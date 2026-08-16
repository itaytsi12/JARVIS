"""Evidence-based curation helpers; no model self-rating is used."""
from .schema import QualityLabel
POSITIVE={QualityLabel.VERIFIED_SUCCESS.value,QualityLabel.USER_APPROVED.value,QualityLabel.CORRECTED.value}
NEGATIVE={QualityLabel.FAILED.value,QualityLabel.REGRESSION.value,QualityLabel.ROLLED_BACK.value,QualityLabel.USER_REJECTED.value}
def is_positive(example):return bool(example.get("training_eligible")) and example.get("quality_label") in POSITIVE
