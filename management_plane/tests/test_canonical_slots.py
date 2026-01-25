from app.services import canonical_slots


def test_action_slot_serialization():
    text = canonical_slots.serialize_action_slot("read", "user")
    assert text == "action=read | actor_type=user"


def test_resource_slot_serialization_skips_none():
    text = canonical_slots.serialize_resource_slot("database", resource_location="cloud")
    assert text == "resource_type=database | resource_location=cloud"


def test_data_slot_formats_bool():
    text = canonical_slots.serialize_data_slot("confidential", pii=False, volume="single")
    assert text == "sensitivity=confidential | pii=false | volume=single"


def test_risk_slot_serialization():
    assert canonical_slots.serialize_risk_slot("required") == "authn=required"
