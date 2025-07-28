# coding: utf-8

"""
Lightweight Python implementation of the Run 2 MET phi (type II) correction, following
https://lathomas.web.cern.ch/lathomas/METStuff/XYCorrections/XYMETCorrection_withUL17andUL18andUL16.h,
taken from https://twiki.cern.ch/twiki/bin/view/CMS/MissingETRun2Corrections?rev=72.

Downloadable source at https://mrieger.web.cern.ch/snippets/met_phi_correction.py,
gist at https://gist.github.com/riga/951076bcde6ea7cb4da2bf7c7417379a.

Supports Python 3.6+. Requires numpy and scipy.

Usage example:

    from met_phi_correction import METPhiCorrector, Campaign
    corrector = METPhiCorrector(
        campaign=Campaign.UL_2017,
        is_data=False,
        is_puppi=False,
    )
    # supports scalar values and arrays, applies broadcasting when mixed
    corr_pt, corr_phi = corrector(
        uncorr_pt=35.0,
        uncorr_phi=-0.5,
        npv=15,
    )

Marcel Rieger
"""

import enum
from typing import Union, Tuple

import numpy as np
import scipy.sparse


class Campaign(enum.Enum):

    PL_2016 = 1
    PL_2017 = 2
    PL_2018 = 3
    UL_2016 = 4  # data only
    UL_2016_APV = 5  # mc only
    UL_2016_NONAPV = 6  # mc only
    UL_2017 = 7
    UL_2018 = 8

    def __int__(self) -> int:
        return self.value

    def validate(self, is_data: bool) -> None:
        if is_data and self in (self.UL_2016_APV, self.UL_2016_NONAPV):
            raise ValueError(f"campaign '{self}' is not supported for data")
        if not is_data and self == self.UL_2016:
            raise ValueError(f"campaign '{self}' is not supported for mc")


class RunEra(enum.Enum):

    A = 1
    B = 2
    C = 3
    D = 4
    E = 5
    F = 6
    F_LATE = 7
    G = 8
    H = 9

    def __int__(self) -> int:
        return self.value


class METPhiCorrector(object):

    def __init__(
        self,
        campaign: Campaign,
        is_data: bool,
        is_puppi: bool = False,
    ):
        super().__init__()

        # lazily verify arguments and set attributes
        assert isinstance(campaign, Campaign)
        campaign.validate(is_data)
        self.campaign = campaign
        self.is_data = bool(is_data)
        self.is_puppi = bool(is_puppi)

        # fixed recommendations
        self.use_metv2 = self.campaign == Campaign.PL_2017

        # create the lookup table
        self.lut = self._create_lookup_table(
            self.campaign,
            self.is_data,
            self.is_puppi,
            self.use_metv2,
        )

    def __call__(
        self,
        uncor_pt: Union[float, np.ndarray],
        uncor_phi: Union[float, np.ndarray],
        npv: Union[int, np.ndarray],
        run: Union[int, np.ndarray] = 0,
    ) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray]]:
        # broadcast inputs to arrays
        all_float_inputs = (
            isinstance(uncor_pt, (int, float)) and
            isinstance(uncor_phi, (int, float)) and
            isinstance(npv, (int, float)) and
            isinstance(run, (int, float))
        )
        uncor_pt, uncor_phi, npv, run, _ = np.broadcast_arrays(
            uncor_pt,
            uncor_phi,
            npv,
            run,
            np.empty(1),
        )

        # clip npv
        npv = np.clip(npv, 1, 100)

        # get the run era
        if self.is_data:
            assert not np.any(run <= 0)
            run_era = self._get_data_run_era(self.campaign, run)
        else:
            run_era = np.zeros(len(run), dtype=np.int32)

        # get the correction factors
        factors = np.array(self.lut[run_era].todense())

        # get corrected px and py values
        corr_px = uncor_pt * np.cos(uncor_phi) - (factors[:, 0] * npv + factors[:, 1])
        corr_py = uncor_pt * np.sin(uncor_phi) - (factors[:, 2] * npv + factors[:, 3])

        # compute results
        corr_pt = (corr_px**2 + corr_py**2)**0.5
        corr_phi = np.arctan2(corr_py, corr_px)

        if all_float_inputs:
            return corr_pt[0], corr_phi[0]
        return corr_pt, corr_phi

    @classmethod
    def _create_lookup_table(
        cls,
        campaign: Campaign,
        is_data: bool,
        is_puppi: bool,
        use_metv2: bool,
    ) -> scipy.sparse.lil_matrix:
        key = (
            "data" if is_data else "mc",
            "puppi" if is_puppi else "no_puppi",
            "metv2" if use_metv2 else "metv1",
        )
        factors = correction_factors[key]

        # get maximum run era to determine the lut shape
        max_run_era = max(int(r) for c, r in factors.keys() if c == campaign)

        # create and fill the table
        lut = scipy.sparse.lil_matrix((max_run_era + 1, 4), dtype=np.float32)
        for (c, r), coeffs in factors.items():
            if c == campaign:
                lut[int(r)] = coeffs

        return lut

    @classmethod
    def _get_data_run_era(
        cls,
        campaign: Campaign,
        run: np.ndarray,
    ) -> np.ndarray:
        run_era = np.zeros(len(run), dtype=np.int32)

        if campaign in (Campaign.PL_2016, Campaign.UL_2016):
            run_era[(run >= 272007) & (run <= 275376)] = int(RunEra.B)
            run_era[(run >= 275657) & (run <= 276283)] = int(RunEra.C)
            run_era[(run >= 276315) & (run <= 276811)] = int(RunEra.D)
            run_era[(run >= 276831) & (run <= 277420)] = int(RunEra.E)
            # small distinction in UL
            if campaign == Campaign.PL_2016:
                run_era[(run >= 277772) & (run <= 278808)] = int(RunEra.F)
            else:  # UL_2016
                run_era[((run >= 277772) & (run <= 278768)) | (run == 278770)] = int(RunEra.F)
                run_era[((run >= 278801) & (run <= 278808)) | (run == 278769)] = int(RunEra.F_LATE)
            run_era[(run >= 278820) & (run <= 280385)] = int(RunEra.G)
            run_era[(run >= 280919) & (run <= 284044)] = int(RunEra.H)

        elif campaign in (Campaign.PL_2017, Campaign.UL_2017):
            run_era[(run >= 297020) & (run <= 299329)] = int(RunEra.B)
            run_era[(run >= 299337) & (run <= 302029)] = int(RunEra.C)
            run_era[(run >= 302030) & (run <= 303434)] = int(RunEra.D)
            run_era[(run >= 303435) & (run <= 304826)] = int(RunEra.E)
            run_era[(run >= 304911) & (run <= 306462)] = int(RunEra.F)

        elif campaign in (Campaign.PL_2018, Campaign.UL_2018):
            run_era[(run >= 315252) & (run <= 316995)] = int(RunEra.A)
            run_era[(run >= 316998) & (run <= 319312)] = int(RunEra.B)
            run_era[(run >= 319313) & (run <= 320393)] = int(RunEra.C)
            run_era[(run >= 320394) & (run <= 325273)] = int(RunEra.D)

        return run_era


# all x and y components are corrected by a 1st degree polynomial dependent on the number of
# primary vertices, so store all 2+2 values per configuration, referring to the x and y parameters
correction_factors = {
    ("mc", "no_puppi", "metv1"): {
        # PL
        (Campaign.PL_2016, 0): (-0.195191, -0.170948, -0.0311891, 0.787627),
        (Campaign.PL_2017, 0): (-0.217714, 0.493361, 0.177058, -0.336648),
        (Campaign.PL_2018, 0): (0.296713, -0.141506, 0.115685, 0.0128193),
        # UL
        (Campaign.UL_2016_NONAPV, 0): (-0.153497, -0.231751, 0.00731978, 0.243323),
        (Campaign.UL_2016_APV, 0): (-0.188743, 0.136539, 0.0127927, 0.117747),
        (Campaign.UL_2017, 0): (-0.300155, 1.90608, 0.300213, -2.02232),
        (Campaign.UL_2018, 0): (0.183518, 0.546754, 0.192263, -0.42121),
    },
    ("mc", "no_puppi", "metv2"): {
        # PL
        (Campaign.PL_2016, 0): (-0.159469, -0.407022, -0.0405812, 0.570415),
        (Campaign.PL_2017, 0): (-0.182569, 0.276542, 0.155652, -0.417633),
        (Campaign.PL_2018, 0): (0.299448, -0.13866, 0.118785, 0.0889588),
    },
    ("mc", "puppi", "metv1"): {
        # UL
        (Campaign.UL_2016_NONAPV, 0): (-0.0058341, -0.395049, 0.00971595, -0.101288),
        (Campaign.UL_2016_APV, 0): (-0.0060447, -0.4183, 0.008331, -0.0990046),
        (Campaign.UL_2017, 0): (-0.0102265, -0.446416, 0.0198663, 0.243182),
        (Campaign.UL_2018, 0): (-0.0214557, 0.969428, 0.0167134, 0.199296),
    },
    ("data", "no_puppi", "metv1"): {
        # PL
        (Campaign.PL_2016, RunEra.B): (-0.0478335, -0.108032, 0.125148, 0.355672),
        (Campaign.PL_2016, RunEra.C): (-0.0916985, 0.393247, 0.151445, 0.114491),
        (Campaign.PL_2016, RunEra.D): (-0.0581169, 0.567316, 0.147549, 0.403088),
        (Campaign.PL_2016, RunEra.E): (-0.065622, 0.536856, 0.188532, 0.495346),
        (Campaign.PL_2016, RunEra.F): (-0.0313322, 0.39866, 0.16081, 0.960177),
        (Campaign.PL_2016, RunEra.G): (0.040803, -0.290384, 0.0961935, 0.666096),
        (Campaign.PL_2016, RunEra.H): (0.0330868, -0.209534, 0.141513, 0.816732),
        (Campaign.PL_2017, RunEra.B): (-0.259456, 1.95372, 0.353928, -2.46685),
        (Campaign.PL_2017, RunEra.C): (-0.232763, 1.08318, 0.257719, -1.1745),
        (Campaign.PL_2017, RunEra.D): (-0.238067, 1.80541, 0.235989, -1.44354),
        (Campaign.PL_2017, RunEra.E): (-0.212352, 1.851, 0.157759, -0.478139),
        (Campaign.PL_2017, RunEra.F): (-0.232733, 2.24134, 0.213341, 0.684588),
        (Campaign.PL_2018, RunEra.A): (0.362865, -1.94505, 0.0709085, -0.307365),
        (Campaign.PL_2018, RunEra.B): (0.492083, -2.93552, 0.17874, -0.786844),
        (Campaign.PL_2018, RunEra.C): (0.521349, -1.44544, 0.118956, -1.96434),
        (Campaign.PL_2018, RunEra.D): (0.531151, -1.37568, 0.0884639, -1.57089),
        # UL
        (Campaign.UL_2016, RunEra.B): (-0.0214894, -0.188255, 0.0876624, 0.812885),
        (Campaign.UL_2016, RunEra.C): (-0.032209, 0.067288, 0.113917, 0.743906),
        (Campaign.UL_2016, RunEra.D): (-0.0293663, 0.21106, 0.11331, 0.815787),
        (Campaign.UL_2016, RunEra.E): (-0.0132046, 0.20073, 0.134809, 0.679068),
        (Campaign.UL_2016, RunEra.F): (-0.0543566, 0.816597, 0.114225, 1.17266),
        (Campaign.UL_2016, RunEra.F_LATE): (0.134616, -0.89965, 0.0397736, 1.0385),
        (Campaign.UL_2016, RunEra.G): (0.121809, -0.584893, 0.0558974, 0.891234),
        (Campaign.UL_2016, RunEra.H): (0.0868828, -0.703489, 0.0888774, 0.902632),
        (Campaign.UL_2017, RunEra.B): (-0.211161, 0.419333, 0.251789, -1.28089),
        (Campaign.UL_2017, RunEra.C): (-0.185184, -0.164009, 0.200941, -0.56853),
        (Campaign.UL_2017, RunEra.D): (-0.201606, 0.426502, 0.188208, -0.58313),
        (Campaign.UL_2017, RunEra.E): (-0.162472, 0.176329, 0.138076, -0.250239),
        (Campaign.UL_2017, RunEra.F): (-0.210639, 0.72934, 0.198626, 1.028),
        (Campaign.UL_2018, RunEra.A): (0.263733, -1.91115, 0.0431304, -0.112043),
        (Campaign.UL_2018, RunEra.B): (0.400466, -3.05914, 0.146125, -0.533233),
        (Campaign.UL_2018, RunEra.C): (0.430911, -1.42865, 0.0620083, -1.46021),
        (Campaign.UL_2018, RunEra.D): (0.457327, -1.56856, 0.0684071, -0.928372),
    },
    ("data", "no_puppi", "metv2"): {
        # PL
        (Campaign.PL_2016, RunEra.B): (-0.0374977, 0.00488262, 0.107373, -0.00732239),
        (Campaign.PL_2016, RunEra.C): (-0.0832562, 0.550742, 0.142469, -0.153718),
        (Campaign.PL_2016, RunEra.D): (-0.0400931, 0.753734, 0.127154, 0.0175228),
        (Campaign.PL_2016, RunEra.E): (-0.0409231, 0.755128, 0.168407, 0.126755),
        (Campaign.PL_2016, RunEra.F): (-0.0161259, 0.516919, 0.141176, 0.544062),
        (Campaign.PL_2016, RunEra.G): (0.0583851, -0.0987447, 0.0641427, 0.319112),
        (Campaign.PL_2016, RunEra.H): (0.0706267, -0.13118, 0.127481, 0.370786),
        (Campaign.PL_2017, RunEra.B): (-0.19563, 1.51859, 0.306987, -1.84713),
        (Campaign.PL_2017, RunEra.C): (-0.161661, 0.589933, 0.233569, -0.995546),
        (Campaign.PL_2017, RunEra.D): (-0.180911, 1.23553, 0.240155, -1.27449),
        (Campaign.PL_2017, RunEra.E): (-0.149494, 0.901305, 0.178212, -0.535537),
        (Campaign.PL_2017, RunEra.F): (-0.165154, 1.02018, 0.253794, 0.75776),
        (Campaign.PL_2018, RunEra.A): (0.362642, -1.55094, 0.0737842, -0.677209),
        (Campaign.PL_2018, RunEra.B): (0.485614, -2.45706, 0.181619, -1.00636),
        (Campaign.PL_2018, RunEra.C): (0.503638, -1.01281, 0.147811, -1.48941),
        (Campaign.PL_2018, RunEra.D): (0.520265, -1.20322, 0.143919, -0.979328),
    },
    ("data", "puppi", "metv1"): {
        # UL
        (Campaign.UL_2016, RunEra.B): (-0.00109025, -0.338093, -0.00356058, 0.128407),
        (Campaign.UL_2016, RunEra.C): (-0.00271913, -0.342268, 0.00187386, 0.104),
        (Campaign.UL_2016, RunEra.D): (-0.00254194, -0.305264, -0.00177408, 0.164639),
        (Campaign.UL_2016, RunEra.E): (-0.00358835, -0.225435, -0.000444268, 0.180479),
        (Campaign.UL_2016, RunEra.F): (0.0056759, -0.454101, -0.00962707, 0.35731),
        (Campaign.UL_2016, RunEra.F_LATE): (0.0234421, -0.371298, -0.00997438, 0.0809178),
        (Campaign.UL_2016, RunEra.G): (0.0182134, -0.335786, -0.0063338, 0.093349),
        (Campaign.UL_2016, RunEra.H): (0.015702, -0.340832, -0.00544957, 0.199093),
        (Campaign.UL_2017, RunEra.B): (-0.00382117, -0.666228, 0.0109034, 0.172188),
        (Campaign.UL_2017, RunEra.C): (-0.00110699, -0.747643, -0.0012184, 0.303817),
        (Campaign.UL_2017, RunEra.D): (-0.00141442, -0.721382, -0.0011873, 0.21646),
        (Campaign.UL_2017, RunEra.E): (0.00593859, -0.851999, -0.00754254, 0.245956),
        (Campaign.UL_2017, RunEra.F): (0.00765682, -0.945001, -0.0154974, 0.804176),
        (Campaign.UL_2018, RunEra.A): (-0.0073377, 0.0250294, -0.000406059, 0.0417346),
        (Campaign.UL_2018, RunEra.B): (0.00434261, 0.00892927, 0.00234695, 0.20381),
        (Campaign.UL_2018, RunEra.C): (0.00198311, 0.37026, -0.016127, 0.402029),
        (Campaign.UL_2018, RunEra.D): (0.00220647, 0.378141, -0.0160244, 0.471053),
    },
}


if __name__ == "__main__":
    corrector = METPhiCorrector(Campaign.UL_2017, False)
    print(corrector(35.0, -0.5, [15, 20]))
