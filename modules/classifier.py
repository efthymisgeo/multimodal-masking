from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from slp.modules.embed import PositionalEncoding
from modules.mmdrop import MultimodalMasking
from modules.multimodal import AttentionFuser
from modules.rnn import AttentiveRNN, TokenRNN


class Classifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        encoded_features: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        """Classifier wrapper module

        Stores a Neural Network encoder and adds a classification layer on top.

        Args:
            encoder (nn.Module): [description]
            encoded_features (int): [description]
            num_classes (int): [description]
            dropout (float): Drop probability
        """
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.drop = nn.Dropout(dropout)
        self.clf = nn.Linear(encoded_features, num_classes)

    def forward(self, *args, **kwargs) -> torch.Tensor:
        """Encode inputs using the encoder network and perform classification

        Returns:
            torch.Tensor: [B, *, num_classes] Logits tensor
        """
        encoded: torch.Tensor = self.encoder(*args, **kwargs)  # type: ignore
        out: torch.Tensor = self.drop(encoded)
        out = self.clf(out)

        return out


class RNNSequenceClassifier(Classifier):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
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
    ):
        encoder = AttentiveRNN(
            input_size,
            hidden_size=hidden_size,
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

        super(RNNSequenceClassifier, self).__init__(
            encoder, encoder.out_size, num_classes, dropout=dropout
        )


class RNNTokenSequenceClassifier(Classifier):
    def __init__(
        self,
        num_classes: int,
        vocab_size: Optional[int] = None,
        embeddings_dim: Optional[int] = None,
        embeddings: Optional[np.ndarray] = None,
        embeddings_dropout: float = 0.0,
        finetune_embeddings: bool = False,
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
    ):
        encoder = TokenRNN(
            vocab_size=vocab_size,
            embeddings_dim=embeddings_dim,
            embeddings=embeddings,
            embeddings_dropout=embeddings_dropout,
            finetune_embeddings=finetune_embeddings,
            hidden_size=hidden_size,
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

        super(RNNTokenSequenceClassifier, self).__init__(
            encoder, encoder.out_size, num_classes, dropout=dropout
        )


class RNNLateFusionClassifier(nn.Module):
    def __init__(
        self,
        modality_feature_sizes,
        num_classes,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        packed_sequence=True,
        merge_bi="cat",
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        num_heads=4,
        max_length=-1,
        dropout=0.1,
        nystrom=True,
        num_landmarks=32,
        kernel_size=33,
        use_mmdrop=False,
        p_mmdrop=0.5,
        p_drop_modalities=None,
        mmdrop_mode="hard",
    ):
        super(RNNLateFusionClassifier, self).__init__()
        self.modalities = modality_feature_sizes.keys()
        self.modality_encoders = nn.ModuleDict(
            {
                m: AttentiveRNN(
                    modality_feature_sizes[m],
                    hidden_size=hidden_size,
                    batch_first=batch_first,
                    layers=num_layers,
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
                )
                for m in self.modalities
            }
        )
        self.mmdrop = None

        if use_mmdrop:
            self.mmdrop = MultimodalMasking(
                p=p_mmdrop,
                n_modalities=len(self.modalities),
                p_mod=p_drop_modalities,
                mode=mmdrop_mode,
            )
        self.out_size = sum([e.out_size for e in self.modality_encoders.values()])
        self.clf = nn.Linear(self.out_size, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        encoded = [
            self.modality_encoders[m](inputs[m], lengths[m]) for m in self.modalities
        ]

        if self.mmdrop is not None:
            encoded = self.mmdrop(*encoded)
        fused = torch.cat(encoded, dim=-1)
        fused = self.drop(fused)
        out = self.clf(fused)

        return out


class RNNSymAttnFusionRNNClassifier(nn.Module):
    def __init__(
        self,
        modality_feature_sizes,
        num_classes,
        num_layers=2,
        batch_first=True,
        bidirectional=True,
        packed_sequence=True,
        merge_bi="cat",
        rnn_type="lstm",
        attention=True,
        hidden_size=100,
        num_heads=4,
        max_length=-1,
        dropout=0.1,
        nystrom=True,
        num_landmarks=32,
        kernel_size=33,
        multi_modal_drop="mmdrop",
        p_mmdrop=0.5,
        mmdrop_before_fuse=True,
        mmdrop_after_fuse=True,
        p_drop_modalities=None,
    ):
        super(RNNSymAttnFusionRNNClassifier, self).__init__()
        self.modalities = modality_feature_sizes.keys()
        self.modality_encoders = nn.ModuleDict(
            {
                m: AttentiveRNN(
                    modality_feature_sizes[m],
                    hidden_size=hidden_size,
                    batch_first=batch_first,
                    layers=num_layers,
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
                    return_hidden=True,
                )
                for m in self.modalities
            }
        )
        self.fuser = AttentionFuser(
            proj_sz=hidden_size,
            residual=1,
            return_hidden=True,
            p_dropout=0.1,
            p_mmdrop=p_mmdrop,
            p_drop_modalities=p_drop_modalities,
            multi_modal_drop=multi_modal_drop,
            mmdrop_before_fuse=mmdrop_before_fuse,
            mmdrop_after_fuse=mmdrop_after_fuse,
        )
        self.rnn = AttentiveRNN(
            self.fuser.out_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            layers=num_layers,
            bidirectional=bidirectional,
            merge_bi="cat",
            dropout=0.1,
            rnn_type=rnn_type,
            packed_sequence=packed_sequence,
            attention=attention,
            num_heads=num_heads,
            nystrom=False,
            return_hidden=True,
        )
        self.clf = nn.Sequential(
            # nn.Linear(self.out_size, self.out_size),
            # nn.ReLU(),
            # nn.Dropout(dropout),
            nn.Linear(self.rnn.out_size, num_classes),
        )
        # self.drop = nn.Dropout(dropout)

    def forward(self, inputs, lengths):
        encoded = {
            m: self.modality_encoders[m](inputs[m], lengths[m]) for m in self.modalities
        }

        fused = self.fuser(encoded["text"], encoded["audio"], encoded["visual"])
        fused = self.rnn(fused, lengths["text"])
        fused = fused.mean(dim=1)
        # fused = fused.sum(dim=1)
        out = self.clf(fused)

        return out


class MOSEITextClassifier(RNNSequenceClassifier):
    def forward(self, x, lengths):
        x = x["text"]
        lengths = lengths["text"]

        return super().forward(x, lengths)
