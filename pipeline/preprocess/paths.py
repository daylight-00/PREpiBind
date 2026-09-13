"""Every path the preprocessing notebooks read or write, in one place.

The notebooks are the ones the datasets were actually built with; only their hardcoded absolute
paths were replaced by the names below, so that the chain runs from a checkout.

Set the two external inputs before running:

    PREPIBIND_IEDB_EXPORT   the unzipped IEDB Export v3 mhc_ligand_full.csv (7.7 GB, build of 2025-04-21)
    PREPIBIND_WORK          scratch for intermediates (default: preprocess/work)

Everything else resolves inside the repository.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# --- external input -------------------------------------------------------------------------
IEDB_EXPORT = os.environ.get(
    "PREPIBIND_IEDB_EXPORT", os.path.join(HERE, "input", "mhc_ligand_full.csv")
)

# --- scratch --------------------------------------------------------------------------------
WORK = os.environ.get("PREPIBIND_WORK", os.path.join(HERE, "work"))
DRAFT = os.path.join(WORK, "draft.csv")


def arm_dir(arm):
    """Working directory for one assay arm; the notebooks write their tables here."""
    d = os.path.join(WORK, arm)
    os.makedirs(d, exist_ok=True)
    return d


# Cross-arm inputs: each MS arm reads the merged table of the arm that supplies its negatives.
# The qualitative arm is stored under "full", which is the name its dataset directory carries.
QUALITATIVE_FULL = os.path.join(WORK, "full", "hum_ani_full.csv")
IC50_FULL = os.path.join(WORK, "ic50", "hum_ani_full.csv")

# --- repository data ------------------------------------------------------------------------
DATA = os.path.join(REPO, "data")
DATASET = os.path.join(DATA, "dataset")
MHC_MAPPING = os.path.join(DATA, "mhc_mapping")
UNIQUE_EPITOPES = os.path.join(DATA, "unique_epitope_whole.csv")

# --- MHC sequence stage ---------------------------------------------------------------------
# Stage 0 reads HLA2_IMGT.csv from here. It is NOT in the repository: the IPD-IMGT/HLA alignment is
# fetched rather than redistributed, so run fetch_mhc_alignment.py first and it writes both that
# file and MHC2MSA.csv here. filtered_manual.json and range_final.txt do ship -- they are the hand
# decisions behind the collapse and the domain windows. See the README.
MHC_SRC = os.path.join(HERE, "mhc_sequences")
