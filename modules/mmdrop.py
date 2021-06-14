import random
from typing import List, Optional

import torch
import torch.nn as nn


class HardMultimodalMasking(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        n_modalities: int = 3,
        p_mod: Optional[List[float]] = None,
        masking: bool = True,
        m3_sequential: bool = True,
    ):
        """M^3 layer implementation

        For each sample in a batch mask one of the modalities with probability p.
        When dealing with sequential data it randomly masks an instance at every timestep.

        Args:
            p (float): mask/drop probability, 1-p is the prob to leave the sequence unaffected
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Mask probabilities for each modality
            masking (bool): masking flag variable, when False uses resca;ing trick
            m3_sequential (bool): mask different instances of the sequence for each modality
        """
        super(HardMultimodalMasking, self).__init__()
        self.p = p
        self.n_modalities = n_modalities
        self.masking = masking
        self.m3_sequential = m3_sequential

        self.p_mod = [1.0 / n_modalities for _ in range(n_modalities)]

        if p_mod is not None:
            self.p_mod = p_mod

    def forward(self, *mods):
        """Fast M^3 forward implementation

        Iterate over batch and timesteps and randomly choose modality to mask

        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations

        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        mods = list(mods)

        # List of [B, L, D]

        if self.training:
            if random.random() < self.p:
                # mask different modality for each sample in batch

                if self.m3_sequential: # mask different modality at every timestep
                    bsz, seqlen = mods[0].size(0), mods[0].size(1)
                    p_modal = torch.distributions.categorical.Categorical(
                        torch.tensor(self.p_mod)
                    )
                    m_cat = p_modal.sample((bsz, seqlen)).to(mods[0].device)
                    for m in range(self.n_modalities):
                        mask = torch.where(m_cat == m, 0, 1).unsqueeze(2)
                        mods[m] = mods[m] * mask

                else:
                    for batch in range(mods[0].size(0)):
                        m = random.choices(
                            list(range(self.n_modalities)), weights=self.p_mod, k=1
                        )[0]

                        # m = random.randint(0, self.n_modalities - 1)
                        mask = torch.ones_like(mods[m])
                        mask[batch] = 0.0
                        mods[m] = mods[m] * mask

        # rescaling trick
        if not self.masking:
            if self.p > 0:
                for m in range(len(mods)):
                    keep_prob = 1 - (self.p / self.n_modalities)
                    mods[m] = mods[m] * (1 / keep_prob)

        return mods

    def __repr__(self):
        shout = (
            self.__class__.__name__
            + "("
            + "p_mask="
            + str(self.p)
            + ", masking="
            + str(self.masking)
            + ", sequential="
            + str(self.m3_sequential)
            + ", p_mod="
            + str(self.p_mod)
            + ")"
        )
        return shout


class SoftMultimodalMasking(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        n_modalities: int = 3,
        p_mod: Optional[List[float]] = None,
        masking: bool = True,
        m3_sequential: bool = False,
    ):
        """Soft M^3 implementation

        Mask p * 100 % of features of a specific modality over batch and timesteps

        Args:
            p (float): mask/drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
            masking: masking flag variable
            m3_sequential: use per timestep masking
        """
        super(SoftMultimodalMasking, self).__init__()
        self.p = p  # p_mask
        self.n_modalities = n_modalities
        self.masking = masking
        self.m3_sequential = m3_sequential

        self.p_mod = [1.0 / n_modalities for _ in range(n_modalities)]

        if p_mod is not None:
            self.p_mod = p_mod

    def forward(self, *mods):
        """Soft M^3 forward

        Sample a binomial mask to mask a random modality in this batch

        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations

        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped
        """
        mods = list(mods)

        if self.training:
            if self.m3_sequential:
                for timestep in range(mods[0].size(1)):
                    m = random.choices(
                        list(range(self.n_modalities)), weights=self.p_mod, k=1
                    )[0]
                    binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                    mods[m][timestep] = mods[m][timstep] * binomial.sample(
                        mods[m][timestep].size()
                    ).to(mods[m].device)
            else:
                # m = random.randint(0, self.n_modalities - 1)
                m = random.choices(
                    list(range(self.n_modalities)), weights=self.p_mod, k=1
                )[0]

                binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
                mods[m] = mods[m] * binomial.sample(mods[m].size()).to(mods[m].device)

        if not self.masking:
            for m in range(self.n_modalities):
                mods[m] = mods[m] * (1.0 / (1 - self.p / self.n_modalities))

        return mods


class MultimodalMasking(nn.Module):
    def __init__(
        self,
        p: float = 0.5,
        n_modalities: int = 3,
        p_mod: Optional[List[float]] = None,
        mode: str = "hard",
        masking: bool = True,
        m3_sequential: bool = True,
    ):
        """M^3 wrapper class

        Mask p * 100 % of features of a specific modality over batch

        Args:
            p (float): mask/drop probability
            n_modalities (int): number of modalities
            p_mod (Optional[List[float]]): Drop probabilities for each modality
            mode (str): Hard or soft M3
            masking (bool): use M3 with no re-scaling
            m3_sequential (bool): per timestep modality masking
        """
        super(MultimodalMasking, self).__init__()

        assert mode in [
            "hard",
            "soft",
        ], "Allowed mode for MultimodalMasking ['hard' | 'soft']"

        if mode == "hard":
            self.m3 = HardMultimodalMasking(
                p=p,
                n_modalities=n_modalities,
                p_mod=p_mod,
                masking=masking,
                m3_sequential=m3_sequential,
            )
        else:
            self.m3 = SoftMultimodalMasking(  # type: ignore
                p=p,
                n_modalities=n_modalities,
                p_mod=p_mod,
                masking=masking,
                m3_sequential=m3_sequential,
            )

    def forward(self, *mods):
        """M^3 wrapper forward

        Perform hard or soft M^3

        Args:
            mods (varargs torch.Tensor): [B, L, D_m] Modality representations

        Returns:
            (List[torch.Tensor]): The modality representations. Some of them are dropped

        """
        return self.m3(*mods)
