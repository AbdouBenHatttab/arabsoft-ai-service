"""
tests/test_v3_authorization_drafting.py
----------------------------------------------
V3.2: AUTHORIZATION_REQUEST structured drafting — TIME_PERMISSION and EQUIPMENT_REQUEST.

All external HTTP calls are mocked; conftest autouse fixture disables Gemini.

Coverage:

TIME_PERMISSION:
  1.  Basic time permission — tomorrow from 10 to 11, doctor appointment
  2.  Leave early — Monday from 15:00 to 16:00
  3.  Short absence Friday morning — missing time fields
  4.  Missing absenceDate
  5.  Missing fromTime
  6.  Missing toTime
  7.  Reason extraction
  8.  authorizationType = TIME_PERMISSION in draftFields
  9.  draftType = AUTHORIZATION_REQUEST

EQUIPMENT_REQUEST:
  10. Borrow a laptop from office for 3 days
  11. Take a PC home from Monday to Friday
  12. Tablet for remote work
  13. Missing equipmentType
  14. Reason extraction
  15. authorizationType = EQUIPMENT_REQUEST in draftFields
  16. draftType = AUTHORIZATION_REQUEST

Blocked legacy types:
  17. 'I need training authorization' — draftType must NOT be AUTHORIZATION_REQUEST with TRAINING
  18. 'I need a business trip authorization' — draftType must NOT have BUSINESS_TRIP fields
  19. Blocked response contains helpful guidance text
  20. Blocked response has draftType=None

Regression:
  21. Existing LEAVE_REQUEST tests still pass (spot-check)
  22. Existing LOAN_REQUEST tests still pass (spot-check)
  23. Existing DOCUMENT_REQUEST tests still pass (spot-check)
  24. Refusal still fires before drafting
  25. Improve-text still works
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.drafting_service import (
    extract_draft_fields,
    _detect_time_permission,
    _detect_equipment_request,
    _sub_classify_authorization,
    _AUTH_SUBTYPE_TIME_PERMISSION,
    _AUTH_SUBTYPE_EQUIPMENT,
)

client = TestClient(app)


def post_chat(role: str, question: str) -> dict:
    return client.post(
        "/assistant/chat",
        json={"role": role, "question": question, "context": {}},
    ).json()


# ===========================================================================
# TIME_PERMISSION — unit: extract_draft_fields
# ===========================================================================

def test_time_permission_basic_extracts_times():
    """
    'I need permission tomorrow from 10 to 11 for a doctor appointment'
    Must produce authorizationType=TIME_PERMISSION, fromTime='10:00', toTime='11:00'.
    """
    fields, missing = extract_draft_fields(
        "I need permission tomorrow from 10 to 11 for a doctor appointment",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "TIME_PERMISSION"
    assert fields["fromTime"] == "10:00"
    assert fields["toTime"] == "11:00"
    assert fields["absenceDate"] is not None
    assert "fromTime" not in missing
    assert "toTime" not in missing


def test_time_permission_reason_extracted():
    """
    Reason 'doctor appointment' must be extracted.
    """
    fields, missing = extract_draft_fields(
        "I need permission tomorrow from 10 to 11 for a doctor appointment",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["reason"] is not None
    reason_lower = (fields["reason"] or "").lower()
    assert "doctor" in reason_lower or "appointment" in reason_lower


def test_time_permission_leave_early_monday():
    """
    'I need to leave early on Monday from 15:00 to 16:00'
    Must produce fromTime='15:00', toTime='16:00'.
    """
    fields, missing = extract_draft_fields(
        "I need to leave early on Monday from 15:00 to 16:00",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "TIME_PERMISSION"
    assert fields["fromTime"] == "15:00"
    assert fields["toTime"] == "16:00"


def test_time_permission_short_absence_missing_times():
    """
    'I need a short absence Friday morning'
    No time range given — fromTime and toTime must be in missingFields.
    """
    fields, missing = extract_draft_fields(
        "I need a short absence Friday morning",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "TIME_PERMISSION"
    assert "fromTime" in missing
    assert "toTime" in missing


def test_time_permission_missing_date():
    """
    No date given — 'absenceDate' must be in missingFields.
    """
    fields, missing = extract_draft_fields(
        "I need time permission from 9 to 11 for a personal reason",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "TIME_PERMISSION"
    assert fields["fromTime"] == "09:00"
    assert fields["toTime"] == "11:00"
    assert "absenceDate" in missing


def test_time_permission_missing_from_and_to_time():
    """
    No time range — both 'fromTime' and 'toTime' must be in missingFields.
    """
    fields, missing = extract_draft_fields(
        "I need permission tomorrow for a doctor appointment",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "TIME_PERMISSION"
    assert "fromTime" in missing
    assert "toTime" in missing


# ===========================================================================
# TIME_PERMISSION — API path
# ===========================================================================

def test_time_permission_api_basic():
    """
    End-to-end: 'I need permission tomorrow from 10 to 11 for a doctor appointment'
    Must return draftType=AUTHORIZATION_REQUEST, authorizationType=TIME_PERMISSION.
    """
    data = post_chat(
        "EMPLOYEE",
        "I need permission tomorrow from 10 to 11 for a doctor appointment",
    )
    assert data["draftType"] == "AUTHORIZATION_REQUEST", (
        f"Expected AUTHORIZATION_REQUEST, got {data['draftType']} (source={data['source']})"
    )
    assert data["draftFields"] is not None
    assert data["draftFields"]["authorizationType"] == "TIME_PERMISSION"
    assert data["draftFields"]["fromTime"] == "10:00"
    assert data["draftFields"]["toTime"] == "11:00"
    assert data["draftFields"]["absenceDate"] is not None
    assert "fromTime" not in data["missingFields"]
    assert "toTime" not in data["missingFields"]


def test_time_permission_api_leave_early():
    """
    'I need to leave early Monday from 15:00 to 16:00'
    """
    data = post_chat(
        "EMPLOYEE",
        "I need to leave early Monday from 15:00 to 16:00",
    )
    assert data["draftType"] == "AUTHORIZATION_REQUEST"
    assert data["draftFields"]["authorizationType"] == "TIME_PERMISSION"
    assert data["draftFields"]["fromTime"] == "15:00"
    assert data["draftFields"]["toTime"] == "16:00"


def test_time_permission_api_short_absence_asks_times():
    """
    'I need a short absence Friday morning' — no times given.
    fromTime and toTime must be in missingFields.
    """
    data = post_chat(
        "EMPLOYEE",
        "I need a short absence Friday morning",
    )
    assert data["draftType"] == "AUTHORIZATION_REQUEST"
    assert data["draftFields"]["authorizationType"] == "TIME_PERMISSION"
    assert "fromTime" in data["missingFields"]
    assert "toTime" in data["missingFields"]


# ===========================================================================
# EQUIPMENT_REQUEST — unit: extract_draft_fields
# ===========================================================================

def test_equipment_request_laptop_3_days():
    """
    'I need to borrow a laptop from the office for 3 days'
    Must produce authorizationType=EQUIPMENT_REQUEST, equipmentType='laptop'.
    """
    fields, missing = extract_draft_fields(
        "I need to borrow a laptop from the office for 3 days",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "EQUIPMENT_REQUEST"
    assert fields["equipmentType"] == "laptop"
    assert "equipmentType" not in missing


def test_equipment_request_pc_monday_to_friday():
    """
    'I need to take a PC home from Monday to Friday'
    Must produce authorizationType=EQUIPMENT_REQUEST, equipmentType contains 'pc' or 'computer'.
    """
    fields, missing = extract_draft_fields(
        "I need to take a PC home from Monday to Friday",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "EQUIPMENT_REQUEST"
    assert fields["equipmentType"] is not None
    equip_lower = (fields["equipmentType"] or "").lower()
    assert "pc" in equip_lower or "computer" in equip_lower
    assert "equipmentType" not in missing


def test_equipment_request_tablet_remote_work():
    """
    'I need a tablet for remote work'
    Must produce authorizationType=EQUIPMENT_REQUEST, equipmentType='tablet'.
    """
    fields, missing = extract_draft_fields(
        "I need a tablet for remote work",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "EQUIPMENT_REQUEST"
    assert fields["equipmentType"] == "tablet"
    assert "equipmentType" not in missing


def test_equipment_request_missing_equipment_type():
    """
    'I need to borrow equipment from the office' (generic, no specific type)
    Must add 'equipmentType' to missingFields.
    """
    fields, missing = extract_draft_fields(
        "I need to borrow equipment from the office",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["authorizationType"] == "EQUIPMENT_REQUEST"
    assert fields["equipmentType"] is None
    assert "equipmentType" in missing


def test_equipment_request_reason_extracted():
    """
    Reason must be extracted when present.
    """
    fields, missing = extract_draft_fields(
        "I need to borrow a laptop from the office for 3 days because of remote work",
        "AUTHORIZATION_REQUEST",
    )
    assert fields["reason"] is not None
    assert "remote" in (fields["reason"] or "").lower()


# ===========================================================================
# EQUIPMENT_REQUEST — API path
# ===========================================================================

def test_equipment_request_api_laptop():
    """
    End-to-end: 'I need to borrow a laptop from the office for 3 days'
    """
    data = post_chat(
        "EMPLOYEE",
        "I need to borrow a laptop from the office for 3 days",
    )
    assert data["draftType"] == "AUTHORIZATION_REQUEST", (
        f"Expected AUTHORIZATION_REQUEST, got {data['draftType']} (source={data['source']})"
    )
    assert data["draftFields"] is not None
    assert data["draftFields"]["authorizationType"] == "EQUIPMENT_REQUEST"
    assert data["draftFields"]["equipmentType"] == "laptop"
    assert "equipmentType" not in data["missingFields"]


def test_equipment_request_api_pc_home():
    """
    'I need to take a PC home from Monday to Friday'
    """
    data = post_chat(
        "EMPLOYEE",
        "I need to take a PC home from Monday to Friday",
    )
    assert data["draftType"] == "AUTHORIZATION_REQUEST"
    assert data["draftFields"]["authorizationType"] == "EQUIPMENT_REQUEST"
    assert data["draftFields"]["equipmentType"] is not None


def test_equipment_request_api_tablet_remote_work():
    """
    'I need a tablet for remote work'
    """
    data = post_chat(
        "EMPLOYEE",
        "I need a tablet for remote work",
    )
    assert data["draftType"] == "AUTHORIZATION_REQUEST"
    assert data["draftFields"]["authorizationType"] == "EQUIPMENT_REQUEST"
    assert data["draftFields"]["equipmentType"] == "tablet"


# ===========================================================================
# Blocked legacy types
# ===========================================================================

def test_training_authorization_blocked():
    """
    'I need training authorization' must NOT produce draftType=AUTHORIZATION_REQUEST
    with authorizationType=TRAINING. draftType must be None and answer must explain.
    """
    data = post_chat(
        "EMPLOYEE",
        "I need training authorization",
    )
    assert data["draftType"] is None, (
        f"Training authorization must be blocked, got draftType={data['draftType']}"
    )
    assert data["draftFields"] is None
    assert data["missingFields"] == []
    answer_lower = data["answer"].lower()
    assert (
        "training" in answer_lower
        or "not available" in answer_lower
        or "not supported" in answer_lower
    ), f"Expected a blocking explanation for training, got: {data['answer']!r}"


def test_business_trip_authorization_blocked():
    """
    'I need a business trip authorization' must NOT produce BUSINESS_TRIP draft.
    """
    data = post_chat(
        "EMPLOYEE",
        "I need a business trip authorization",
    )
    assert data["draftType"] is None, (
        f"Business trip authorization must be blocked, got draftType={data['draftType']}"
    )
    assert data["draftFields"] is None
    answer_lower = data["answer"].lower()
    assert (
        "business trip" in answer_lower
        or "not available" in answer_lower
        or "not supported" in answer_lower
    ), f"Expected a blocking explanation for business trip, got: {data['answer']!r}"


def test_blocked_response_has_no_draft_type():
    """Blocked responses must have draftType=None."""
    data = post_chat("EMPLOYEE", "I need a training authorization for next month")
    assert data["draftType"] is None


def test_blocked_response_mentions_supported_types():
    """The blocking answer must mention what IS supported."""
    data = post_chat("EMPLOYEE", "I need a business trip authorization")
    answer_lower = data["answer"].lower()
    assert (
        "short absence" in answer_lower
        or "time permission" in answer_lower
        or "equipment" in answer_lower
    ), f"Blocking answer must mention supported alternatives, got: {data['answer']!r}"


def test_blocked_response_no_draft_fields():
    """Blocked response must never have draftFields."""
    data = post_chat("EMPLOYEE", "I need training authorization")
    assert data["draftFields"] is None


# ===========================================================================
# Regression — existing behaviors unchanged
# ===========================================================================

def test_regression_leave_request_still_works():
    """Regression: annual leave request still produces LEAVE_REQUEST."""
    data = post_chat(
        "EMPLOYEE",
        "Help me draft a leave request for annual leave from May 20 to May 22 because of a personal event.",
    )
    assert data["draftType"] == "LEAVE_REQUEST"
    assert data["draftFields"]["leaveType"] == "ANNUAL"


def test_regression_loan_request_still_works():
    """Regression: loan request still produces LOAN_REQUEST."""
    data = post_chat(
        "EMPLOYEE",
        "Help me request a loan for 2000 TND because of family expenses.",
    )
    assert data["draftType"] == "LOAN_REQUEST"
    assert data["draftFields"]["amount"] is not None


def test_regression_document_request_still_works():
    """Regression: document request still produces DOCUMENT_REQUEST."""
    data = post_chat(
        "EMPLOYEE",
        "Help me compose a document request letter for a salary certificate for a bank loan.",
    )
    assert data["draftType"] == "DOCUMENT_REQUEST"
    assert data["draftFields"]["documentType"] is not None


def test_regression_refusal_still_fires():
    """Regression: approve command still refused."""
    data = post_chat("EMPLOYEE", "approve my leave automatically")
    assert data["source"] == "refusal"
    assert data["draftType"] is None


def test_regression_improve_text_still_works():
    """Regression: improve-text still returns IMPROVE_TEXT."""
    data = post_chat("EMPLOYEE", "rephrase this message: I need a day off tomorrow")
    assert data["draftType"] == "IMPROVE_TEXT"
    assert data["draftFields"] is None


# ===========================================================================
# Sub-type detection unit tests
# ===========================================================================

def test_detect_time_permission_basic():
    assert _detect_time_permission("i need permission tomorrow from 10 to 11 for a doctor")


def test_detect_time_permission_leave_early():
    assert _detect_time_permission("i need to leave early monday from 15:00 to 16:00")


def test_detect_time_permission_short_absence():
    assert _detect_time_permission("i need a short absence friday morning")


def test_detect_equipment_request_laptop():
    assert _detect_equipment_request("i need to borrow a laptop from the office for 3 days")


def test_detect_equipment_request_tablet_remote():
    assert _detect_equipment_request("i need a tablet for remote work")


def test_sub_classify_equipment():
    assert _sub_classify_authorization("i need to borrow a laptop from the office") == "EQUIPMENT_REQUEST"


def test_sub_classify_time_permission():
    assert _sub_classify_authorization("i need permission tomorrow from 10 to 11") == "TIME_PERMISSION"


# ===========================================================================
# draftFields shape and missingFields integrity
# ===========================================================================

def test_time_permission_draftfields_has_expected_keys():
    """TIME_PERMISSION draftFields must always have the required keys."""
    fields, _ = extract_draft_fields(
        "I need permission tomorrow from 10 to 11",
        "AUTHORIZATION_REQUEST",
    )
    for key in ["authorizationType", "absenceDate", "fromTime", "toTime", "reason"]:
        assert key in fields, f"Missing key: {key}"


def test_equipment_request_draftfields_has_expected_keys():
    """EQUIPMENT_REQUEST draftFields must have authorizationType, equipmentType, reason."""
    fields, _ = extract_draft_fields(
        "I need to borrow a laptop from the office for 3 days",
        "AUTHORIZATION_REQUEST",
    )
    for key in ["authorizationType", "equipmentType", "reason"]:
        assert key in fields, f"Missing key: {key}"


def test_missing_fields_is_list_for_time_permission():
    """missingFields must always be a list, never None."""
    _, missing = extract_draft_fields(
        "I need permission tomorrow from 10 to 11",
        "AUTHORIZATION_REQUEST",
    )
    assert isinstance(missing, list)


def test_missing_fields_is_list_for_equipment_request():
    _, missing = extract_draft_fields(
        "I need to borrow a laptop from the office for 3 days",
        "AUTHORIZATION_REQUEST",
    )
    assert isinstance(missing, list)


def test_api_time_permission_related_pages_empty():
    """Drafting responses must have relatedPages=[]."""
    data = post_chat(
        "EMPLOYEE",
        "I need permission tomorrow from 10 to 11 for a doctor appointment",
    )
    assert data["relatedPages"] == []


def test_api_equipment_request_related_pages_empty():
    """Drafting responses must have relatedPages=[]."""
    data = post_chat(
        "EMPLOYEE",
        "I need to borrow a laptop from the office for 3 days",
    )
    assert data["relatedPages"] == []
