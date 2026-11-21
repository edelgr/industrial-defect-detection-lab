"""JSON schemas for research catalog records (datasets, algorithms, publications)."""

from .common import (ANY, NULLABLE_INT, NULLABLE_STR, NULLABLE_STR_LIST, NULLABLE_URL,
                     STR_LIST, VERIFICATION_STATUS)

PUBLICATION_SCHEMA = {
    "type": "object",
    "required": ["id", "title", "authors", "year", "venue", "doi", "arxiv_id", "paper_url",
                 "pdf_url", "code_url", "datasets", "algorithms", "verification", "bibtex"],
    "properties": {
        "id": {"type": "string", "pattern": r"^[a-z0-9_]+$"},
        "title": {"type": "string", "minLength": 3},
        "authors": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "year": {"type": "integer", "minimum": 1900, "maximum": 2100},
        "venue": {"type": "string"},
        "doi": NULLABLE_STR,
        "arxiv_id": NULLABLE_STR,
        "paper_url": NULLABLE_URL,
        "pdf_url": NULLABLE_URL,
        "code_url": NULLABLE_URL,
        "code_is_official": {"type": ["boolean", "null"]},
        "datasets": STR_LIST,
        "algorithms": STR_LIST,
        "verification": {
            "type": "object",
            "required": ["status", "date"],
            "properties": {"status": VERIFICATION_STATUS, "date": NULLABLE_STR, "notes": NULLABLE_STR},
        },
        "bibtex": {"type": "string", "pattern": r"^\s*@"},
    },
    "additionalProperties": False,
}

DATASET_FIELDS = [
    "id", "name", "acronym", "description_en", "description_es", "scientific_purpose",
    "year_first_release", "publication_year", "authors", "institutions", "publication_title",
    "publication_venue", "doi", "paper_url", "paper_pdf_url", "official_website",
    "official_repository_url", "implementation_repository_url", "download_url", "access_method",
    "license", "license_url", "redistribution_allowed", "modalities", "categories",
    "sample_count", "anomaly_types", "annotation_types", "data_formats", "image_resolution",
    "train_split", "validation_split", "test_split", "ground_truth_availability",
    "benchmark_protocol", "principal_challenges", "scientific_contribution", "limitations",
    "recommended_educational_exercises", "installation_status", "source_verification_date",
    "source_verification_status",
]

DATASET_SCHEMA = {
    "type": "object",
    "required": DATASET_FIELDS + ["publication_ids", "resource_guide"],
    "properties": {
        "id": {"type": "string", "pattern": r"^D\d{2}$"},
        "name": {"type": "string"},
        "acronym": {"type": "string"},
        "description_en": {"type": "string", "minLength": 20},
        "description_es": {"type": "string", "minLength": 20},
        "scientific_purpose": {"type": "string"},
        "year_first_release": NULLABLE_INT,
        "publication_year": NULLABLE_INT,
        "authors": NULLABLE_STR_LIST,
        "institutions": NULLABLE_STR_LIST,
        "publication_title": NULLABLE_STR,
        "publication_venue": NULLABLE_STR,
        "doi": NULLABLE_STR,
        "paper_url": NULLABLE_URL,
        "paper_pdf_url": NULLABLE_URL,
        "official_website": NULLABLE_URL,
        "official_repository_url": NULLABLE_URL,
        "implementation_repository_url": NULLABLE_URL,
        "download_url": NULLABLE_URL,
        "access_method": NULLABLE_STR,
        "license": NULLABLE_STR,
        "license_url": NULLABLE_URL,
        "redistribution_allowed": {"type": ["boolean", "string", "null"]},
        "modalities": STR_LIST,
        "categories": NULLABLE_STR_LIST,
        "sample_count": {"type": ["integer", "string", "null"]},
        "anomaly_types": STR_LIST,
        "annotation_types": NULLABLE_STR_LIST,
        "data_formats": NULLABLE_STR_LIST,
        "image_resolution": NULLABLE_STR,
        "train_split": NULLABLE_STR,
        "validation_split": NULLABLE_STR,
        "test_split": NULLABLE_STR,
        "ground_truth_availability": NULLABLE_STR,
        "benchmark_protocol": NULLABLE_STR,
        "principal_challenges": NULLABLE_STR_LIST,
        "scientific_contribution": NULLABLE_STR,
        "limitations": NULLABLE_STR_LIST,
        "recommended_educational_exercises": NULLABLE_STR_LIST,
        "installation_status": {"type": "string", "enum": ["not_installed", "partial", "installed"]},
        "source_verification_date": NULLABLE_STR,
        "source_verification_status": VERIFICATION_STATUS,
        "publication_ids": STR_LIST,
        "resource_guide": {"type": "string"},
        "adapter": NULLABLE_STR,
        "size_notes": NULLABLE_STR,
        "source_notes": NULLABLE_STR,
        # complementary references added by the maintainers; bibliographic fields may be null while
        # unverified — the GUI then shows only the link and its verification status
        "complementary_references": {"type": "array", "items": {
            "type": "object", "required": ["url", "label", "verification_status", "note"],
            "properties": {"url": {"type": "string", "pattern": "^https://"}, "label": {"type": "string"},
                           "title": NULLABLE_STR, "authors": NULLABLE_STR_LIST, "venue": NULLABLE_STR,
                           "doi": NULLABLE_STR, "publication_id": NULLABLE_STR,
                           "verification_status": VERIFICATION_STATUS,
                           "note": {"type": "string"}, "added": NULLABLE_STR},
            "additionalProperties": False}},
    },
    "additionalProperties": False,
}

IMPLEMENTATION_STATUS = [
    "explanation_only",
    "runnable_classical",
    "runnable_pretrained",
    "requires_fitting",
    "missing_checkpoint",
    "unavailable_dependencies",
    "validated_local_dataset",
]

IMPLEMENTATION_TYPE = [
    "official",               # code released by the paper's authors
    "third_party_reproduction",
    "anomalib",               # Anomalib (Intel / open-edge-platform) integration
    "library",                # general library implementation (OpenCV, scikit-image ...)
    "idd_lab_educational",    # written for IDD Lab following the paper
    "none",
]

ALGORITHM_FIELDS = [
    "id", "name", "family", "year", "paper_title", "paper_authors", "venue", "doi", "paper_url",
    "paper_pdf_url", "official_repository_url", "implementation_repository_url",
    "implementation_type", "implementation_version", "source_license", "weights_url",
    "weights_license", "learning_paradigm", "training_data_requirements", "supported_modalities",
    "supported_tasks", "inputs", "outputs", "intuitive_explanation_en", "intuitive_explanation_es",
    "scientific_principle", "main_processing_steps", "mathematical_formulation", "advantages",
    "limitations", "failure_cases", "computational_requirements", "recommended_datasets",
    "educational_exercises", "source_verification_date", "implementation_status",
]

ALGORITHM_SCHEMA = {
    "type": "object",
    "required": ALGORITHM_FIELDS + ["group", "publication_ids"],
    "properties": {
        "id": {"type": "string", "pattern": r"^[a-z0-9_]+$"},
        "group": {"type": "string", "enum": list("ABCDEFG")},
        "name": {"type": "string"},
        "family": {"type": "string"},
        "year": NULLABLE_INT,
        "paper_title": NULLABLE_STR,
        "paper_authors": NULLABLE_STR_LIST,
        "venue": NULLABLE_STR,
        "doi": NULLABLE_STR,
        "paper_url": NULLABLE_URL,
        "paper_pdf_url": NULLABLE_URL,
        "official_repository_url": NULLABLE_URL,
        "implementation_repository_url": NULLABLE_URL,
        "implementation_type": {"type": "string", "enum": IMPLEMENTATION_TYPE},
        "implementation_version": NULLABLE_STR,
        "implementation_notes": NULLABLE_STR,
        "source_license": NULLABLE_STR,
        "weights_url": NULLABLE_URL,
        "weights_license": NULLABLE_STR,
        "learning_paradigm": {"type": "string"},
        "training_data_requirements": {"type": "string"},
        "supported_modalities": STR_LIST,
        "supported_tasks": STR_LIST,
        "inputs": {"type": "string"},
        "outputs": {"type": "string"},
        "intuitive_explanation_en": {"type": "string", "minLength": 20},
        "intuitive_explanation_es": {"type": "string", "minLength": 20},
        "scientific_principle": {"type": "string"},
        "main_processing_steps": STR_LIST,
        "mathematical_formulation": NULLABLE_STR,
        "advantages": STR_LIST,
        "limitations": STR_LIST,
        "failure_cases": STR_LIST,
        "computational_requirements": {"type": "string"},
        "recommended_datasets": STR_LIST,
        "educational_exercises": STR_LIST,
        "source_verification_date": NULLABLE_STR,
        "source_verification_status": VERIFICATION_STATUS,
        "implementation_status": {"type": "string", "enum": IMPLEMENTATION_STATUS},
        "publication_ids": STR_LIST,
        "runtime": NULLABLE_STR,     # dotted path "module:Class" of the IDD Lab adapter
        "docs_url": NULLABLE_URL,
        "industrial_use": NULLABLE_STR,
    },
    "additionalProperties": False,
}

__all__ = ["PUBLICATION_SCHEMA", "DATASET_SCHEMA", "ALGORITHM_SCHEMA", "DATASET_FIELDS",
           "ALGORITHM_FIELDS", "IMPLEMENTATION_STATUS", "IMPLEMENTATION_TYPE", "ANY"]
