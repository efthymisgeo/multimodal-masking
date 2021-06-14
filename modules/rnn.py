from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from loguru import logger


class AttentiveRNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        batch_first: bool = True,
        layers: int = 1,
        bidirectional: bool = False,
        merge_bi: str = "cat",
        dropout: float = 0.1,
        rnn_type: str = "lstm",
        packed_sequence: bool = True,
        attention: bool = False,
        max_length: int = -1,
        num_heads: int = 1,
        nystrom: bool = True,
        num_landmarks: int = 32,
        kernel_size: Optional[int] = 33,
        inverse_iterations: int = 6,
        return_hidden: bool = False,
    ):
        """RNN with embedding layer and optional attention mechanism

        Single-headed scaled dot-product attention is used as an attention mechanism

        Args:
            input_size (int): Input features dimension
            hidden_size (int): Hidden features
            batch_first (bool): Use batch first representation type. Defaults to True.
            layers (int): Number of RNN layers. Defaults to 1.
            bidirectional (bool): Use bidirectional RNNs. Defaults to False.
            merge_bi (str): How bidirectional states are merged. Defaults to "cat".
            dropout (float): Dropout probability. Defaults to 0.0.
            rnn_type (str): lstm or gru. Defaults to "lstm".
            packed_sequence (bool): Use packed sequences. Defaults to True.
            max_length (int): Maximum sequence length for fixed length padding. If -1 takes the
                largest sequence length in this batch
            attention (bool): Use attention mechanism. Defaults to False
            num_heads (int): Number of attention heads. If 1 uses single headed attention
            nystrom (bool): Use nystrom approximation for multihead attention
            num_landmarks (int): Number of landmark sequence elements for nystrom attention
            kernel_size (int): Kernel size for multihead attention output residual convolution
            inverse_iterations (int): Number of iterations for moore-penrose inverse approximation
                in nystrom attention. 6 is a good value
            return_hidden (bool): Return all hidden states or a single state. If attention is
                used the weighted mean of hidden states is returned. If no attention is used and
                return_hidden is False the last hidden state is returned. If return_hidden is true
                all hidden states are returned (weighted by attention scores if attention is used).
        """
        super(AttentiveRNN, self).__init__()
        self.rnn = RNN(
            input_size,  # type: ignore
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            merge_bi=merge_bi,
            bidirectional=bidirectional,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            max_length=max_length,
        )
        self.out_size = (
            hidden_size
            if not (bidirectional and merge_bi == "cat")
            else 2 * hidden_size
        )
        self.batch_first = batch_first

        self.attention = None
        self.return_hidden = return_hidden

        if attention:
            if num_heads == 1:
                self.attention = Attention(
                    attention_size=self.out_size, dropout=dropout
                )
            else:
                self.attention = MultiheadAttention(  # type: ignore
                    attention_size=self.out_size,
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    nystrom=nystrom,
                    num_landmarks=num_landmarks,
                    inverse_iterations=inverse_iterations,
                    dropout=dropout,
                )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Attentive RNN forward pass

        If self.attention=True then the outputs are the weighted sum of the RNN hidden states with the attention score weights
        Else the output is the last hidden state of the RNN.

        Args:
            x (torch.Tensor): [B, L] Input token ids
            lengths (torch.Tensor): [B] Original sequence lengths

        Returns:
            torch.Tensor: [B, H] or [B, 2*H] Output features to be used for classification
        """
        out, last_hidden, _ = self.rnn(x, lengths)

        if self.attention is not None:
            out, _ = self.attention(
                out,
                attention_mask=pad_mask(
                    lengths, max_length=out.size(1) if self.batch_first else out.size(0)
                ),
            )

            if self.return_hidden:
                outputs = out
            else:
                outputs = out.mean(dim=1)
                # outputs = out.sum(dim=1)
        else:
            if self.return_hidden:
                outputs = out
            else:
                outputs = last_hidden

        return outputs  # type: ignore


class TokenRNN(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        vocab_size: Optional[int] = None,
        embeddings_dim: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None,
        embeddings_dropout: float = 0.0,
        finetune_embeddings: bool = False,
        batch_first: bool = True,
        layers: int = 1,
        bidirectional: bool = False,
        merge_bi: str = "cat",
        dropout: float = 0.1,
        rnn_type: str = "lstm",
        packed_sequence: bool = True,
        attention: bool = False,
        max_length: int = -1,
        num_heads: int = 1,
        nystrom: bool = True,
        num_landmarks: int = 32,
        kernel_size: Optional[int] = 33,
        inverse_iterations: int = 6,
    ):
        """RNN with embedding layer and optional attention mechanism

        Single-headed scaled dot-product attention is used as an attention mechanism

        Args:
            hidden_size (int): Hidden features
            vocab_size (Optional[int]): Vocabulary size. Defaults to None.
            embeddings_dim (Optional[int]): Embedding dimension. Defaults to None.
            embeddings (Optional[np.ndarray]): Embedding matrix. Defaults to None.
            embeddings_dropout (float): Embedding dropout probability. Defaults to 0.0.
            finetune_embeddings (bool): Finetune embeddings? Defaults to False.
            batch_first (bool): Use batch first representation type. Defaults to True.
            layers (int): Number of RNN layers. Defaults to 1.
            bidirectional (bool): Use bidirectional RNNs. Defaults to False.
            merge_bi (str): How bidirectional states are merged. Defaults to "cat".
            dropout (float): Dropout probability. Defaults to 0.0.
            rnn_type (str): lstm or gru. Defaults to "lstm".
            packed_sequence (bool): Use packed sequences. Defaults to True.
            max_length (int): Maximum sequence length for fixed length padding. If -1 takes the
                largest sequence length in this batch
            attention (bool): Use attention mechanism. Defaults to False
            num_heads (int): Number of attention heads. If 1 uses single headed attention
            nystrom (bool): Use nystrom approximation for multihead attention
            num_landmarks (int): Number of landmark sequence elements for nystrom attention
            kernel_size (int): Kernel size for multihead attention output residual convolution
            inverse_iterations (int): Number of iterations for moore-penrose inverse approximation
                in nystrom attention. 6 is a good value
        """
        super(TokenRNN, self).__init__()

        if embeddings is None:
            finetune_embeddings = True
            assert (
                vocab_size is not None
            ), "You should either pass an embeddings matrix or vocab size"
            assert (
                embeddings_dim is not None
            ), "You should either pass an embeddings matrix or embeddings_dim"
        else:
            vocab_size = embeddings.shape[0]
            embeddings_dim = embeddings.shape[1]

        self.embed = Embed(
            vocab_size,  # type: ignore
            embeddings_dim,  # type: ignore
            embeddings=embeddings,
            dropout=embeddings_dropout,
            scale=hidden_size ** 0.5,
            trainable=finetune_embeddings,
        )
        self.encoder = AttentiveRNN(
            embeddings_dim,  # type: ignore
            hidden_size,
            batch_first=batch_first,
            layers=layers,
            bidirectional=bidirectional,
            merge_bi=merge_bi,
            dropout=dropout,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            attention=attention,
            max_length=max_length,
            num_heads=num_heads,
            nystrom=nystrom,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            inverse_iterations=inverse_iterations,
        )

        self.out_size = self.encoder.out_size

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Token RNN forward pass

        If self.attention=True then the outputs are the weighted sum of the RNN hidden states with the attention score weights
        Else the output is the last hidden state of the RNN.

        Args:
            x (torch.Tensor): [B, L] Input token ids
            lengths (torch.Tensor): [B] Original sequence lengths

        Returns:
            torch.Tensor: [B, H] or [B, 2*H] Output features to be used for classification
        """
        x = self.embed(x)
        out = self.encoder(x, lengths)

        return out  # type: ignore
