"""Serialize validated application rules as W3C ODRL 2.2 JSON-LD.

The local ``ontology/ODRL22.rdf`` file is used as the vocabulary authority for
actions, constraint left operands, and operators. Serialization is deliberately
conservative: an unsupported field raises an error instead of silently losing
policy meaning in a document that merely looks like valid ODRL.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from xml.etree import ElementTree

from app.utils.path_util import PROJECT_ROOT


ODRL_CONTEXT = "http://www.w3.org/ns/odrl.jsonld"
ODRL_NAMESPACE = "http://www.w3.org/ns/odrl/2/"
XSD_NAMESPACE = "http://www.w3.org/2001/XMLSchema#"
RDF_ABOUT = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about"
DEFAULT_ONTOLOGY_PATH = Path(PROJECT_ROOT) / "ontology" / "ODRL22.rdf"

RULE_TYPE_TO_PROPERTY = {
    "permission": "permission",
    "prohibition": "prohibition",
    "obligation": "obligation",
}

# Mappings are used only where the natural-language operation has a clear
# equivalent in the ODRL Common Vocabulary. Unknown actions are rejected.
ACTION_ALIASES = {
    "access": "read",
    "accessed": "read",
    "archive": "archive",
    "anonymise": "anonymize",
    "anonymised": "anonymize",
    "anonymising": "anonymize",
    "anonymize": "anonymize",
    "anonymized": "anonymize",
    "anonymizing": "anonymize",
    "copy": "reproduce",
    "delete": "delete",
    "deleted": "delete",
    "download": "reproduce",
    "downloaded": "reproduce",
    "erase": "delete",
    "execute": "execute",
    "export": "export",
    "modify": "modify",
    "notify": "inform",
    "notified": "inform",
    "notifying": "inform",
    "print": "print",
    "read": "read",
    "retain": "archive",
    "retained": "archive",
    "share": "distribute",
    "shared": "distribute",
    "store": "archive",
    "stored": "archive",
    "transfer": "transfer",
    "use": "use",
    "used": "use",
    "view": "read",
}


class ODRLSerializationError(ValueError):
    """Raised when application rules cannot be represented without loss."""


# Backward-compatible name retained for callers written while serialization
# was a placeholder. It is no longer raised by the implemented serializer.
class ODRLSerializationNotImplementedError(NotImplementedError):
    """Deprecated placeholder exception retained for API compatibility."""


@lru_cache(maxsize=4)
def load_odrl_vocabulary(
    ontology_path: str = str(DEFAULT_ONTOLOGY_PATH),
) -> dict[str, frozenset[str]]:
    """Load standard vocabulary terms from the bundled ODRL 2.2 RDF file."""
    path = Path(ontology_path)
    if not path.is_file():
        raise ODRLSerializationError(f"ODRL ontology was not found: {path}")
    try:
        root = ElementTree.parse(path).getroot()
    except ElementTree.ParseError as exc:
        raise ODRLSerializationError(
            f"ODRL ontology is not valid RDF/XML: {path}"
        ) from exc

    terms: dict[str, set[str]] = {
        "actions": set(),
        "left_operands": set(),
        "operators": set(),
    }
    tag_to_group = {
        f"{{{ODRL_NAMESPACE}}}Action": "actions",
        f"{{{ODRL_NAMESPACE}}}LeftOperand": "left_operands",
        f"{{{ODRL_NAMESPACE}}}Operator": "operators",
    }
    for element in root.iter():
        group = tag_to_group.get(element.tag)
        iri = element.attrib.get(RDF_ABOUT, "")
        if group and iri.startswith(ODRL_NAMESPACE):
            terms[group].add(iri.removeprefix(ODRL_NAMESPACE))

    required = {
        "actions": {"use", "read", "reproduce", "archive"},
        "left_operands": {
            "count",
            "dateTime",
            "elapsedTime",
            "event",
            "purpose",
            "recipient",
        },
        "operators": {"eq", "lt", "lteq", "gt", "gteq"},
    }
    for group, required_terms in required.items():
        missing = required_terms - terms[group]
        if missing:
            raise ODRLSerializationError(
                f"ODRL ontology is missing required {group}: "
                + ", ".join(sorted(missing))
            )
    return {key: frozenset(values) for key, values in terms.items()}


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _is_absolute_iri(value: str) -> bool:
    parsed = urlsplit(value)
    return bool(parsed.scheme) and not any(character.isspace() for character in value)


def _slug(value: str, fallback: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return (normalized[:48].strip("-") or fallback)


def _mint_iri(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"urn:lexlegal:{kind}:{_slug(value, kind)}:{digest}"


def _as_iri(kind: str, value: Any, *, required: bool) -> str | None:
    text = _clean_text(value)
    if not text:
        if required:
            raise ODRLSerializationError(f"A non-empty {kind} is required.")
        return None
    return text if _is_absolute_iri(text) else _mint_iri(kind, text)


def _canonical_action(value: Any, vocabulary: dict[str, frozenset[str]]) -> str:
    text = _clean_text(value).casefold()
    if not text:
        raise ODRLSerializationError("Every ODRL rule requires an action.")
    if ";" in text or "," in text:
        raise ODRLSerializationError(
            f"Action must be atomic before serialization: {value!r}"
        )
    words = re.findall(r"[a-zA-Z]+", text)
    candidates = [text, *reversed(words)]
    for candidate in candidates:
        mapped = ACTION_ALIASES.get(candidate, candidate)
        if mapped in vocabulary["actions"]:
            return mapped
    raise ODRLSerializationError(
        f"Action {value!r} has no approved ODRL 2.2 vocabulary mapping."
    )


def _typed_value(value: Any, xsd_type: str) -> dict[str, Any]:
    return {"@value": value, "@type": f"xsd:{xsd_type}"}


def _duration_constraint(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"\b(\d+)\s*(second|minute|hour|day|week|month|year)s?\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    amount = int(match.group(1))
    unit = match.group(2).casefold()
    if amount <= 0:
        raise ODRLSerializationError("Duration must be greater than zero.")
    designator = {
        "second": f"PT{amount}S",
        "minute": f"PT{amount}M",
        "hour": f"PT{amount}H",
        "day": f"P{amount}D",
        "week": f"P{amount * 7}D",
        "month": f"P{amount}M",
        "year": f"P{amount}Y",
    }[unit]
    operator = "gteq" if re.search(r"\bat least\b", text, re.I) else "lteq"
    if re.search(r"\b(exactly|equal to)\b", text, re.I):
        operator = "eq"
    return {
        "leftOperand": "elapsedTime",
        "operator": operator,
        "rightOperand": _typed_value(designator, "duration"),
    }


def _count_constraint(text: str) -> dict[str, Any] | None:
    normalized = text.casefold()
    amount: int | None = None
    if re.search(r"\b(once|one[- ]time|one time)\b", normalized):
        amount = 1
    else:
        match = re.search(r"\b(\d+)\s*times?\b", normalized)
        if match:
            amount = int(match.group(1))
    if amount is None:
        return None
    operator = "gteq" if "at least" in normalized else "lteq"
    if "exactly" in normalized:
        operator = "eq"
    return {
        "leftOperand": "count",
        "operator": operator,
        "rightOperand": _typed_value(amount, "integer"),
    }


def _absolute_date_constraint(text: str) -> dict[str, Any] | None:
    """Map a calendar deadline to the ODRL dateTime left operand."""
    date_value: str | None = None
    patterns = (
        (r"\b\d{4}-\d{2}-\d{2}\b", ("%Y-%m-%d",)),
        (
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4}\b",
            ("%d %B %Y", "%d %b %Y"),
        ),
        (
            r"\b[A-Za-z]+\s+\d{1,2}(?:st|nd|rd|th)?[,]?\s+\d{4}\b",
            ("%B %d %Y", "%b %d %Y"),
        ),
    )
    for pattern, formats in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        candidate = re.sub(
            r"(?<=\d)(?:st|nd|rd|th)\b|,",
            "",
            match.group(0),
            flags=re.IGNORECASE,
        )
        for date_format in formats:
            try:
                date_value = datetime.strptime(
                    candidate.title(),
                    date_format,
                ).date().isoformat()
                break
            except ValueError:
                continue
        if date_value:
            break
    if not date_value:
        return None

    normalized = text.casefold()
    if re.search(
        r"\b(on or after|from|not before|no earlier than)\b",
        normalized,
    ):
        operator = "gteq"
    elif re.search(
        r"\b(until|through|by|on or before|not after|no later than)\b",
        normalized,
    ):
        operator = "lteq"
    elif re.search(r"\b(before|earlier than)\b", normalized):
        operator = "lt"
    elif re.search(r"\b(after|later than)\b", normalized):
        operator = "gt"
    else:
        operator = "eq"
    return {
        "leftOperand": "dateTime",
        "operator": operator,
        "rightOperand": _typed_value(date_value, "date"),
    }


def _recipient_constraint(text: str) -> dict[str, Any] | None:
    """Represent an explicitly named recipient category."""
    match = re.search(
        r"\bto\s+(third\s+part(?:y|ies)|external\s+part(?:y|ies))\b",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    recipient = _clean_text(match.group(1))
    return {
        "leftOperand": "recipient",
        "operator": "eq",
        "rightOperand": {"@id": _mint_iri("party-category", recipient)},
    }


def _event_constraint(text: str) -> dict[str, Any] | None:
    """Represent an explicitly identified event governing rule execution."""
    match = re.search(r"\bwhen\s+(.+)$", text, flags=re.IGNORECASE)
    if not match:
        return None
    event = _clean_text(match.group(1)).rstrip(".,;:")
    if not event:
        return None
    return {
        "leftOperand": "event",
        "operator": "eq",
        "rightOperand": {"@id": _mint_iri("event", event)},
    }


def _serialize_constraints(
    constraint: Any,
    purpose: Any,
    action: str,
    vocabulary: dict[str, frozenset[str]],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    constraint_text = _clean_text(constraint)
    if constraint_text:
        count = _count_constraint(constraint_text)
        duration = _duration_constraint(constraint_text)
        absolute_date = _absolute_date_constraint(constraint_text)
        recipient = _recipient_constraint(constraint_text)
        event = _event_constraint(constraint_text)
        if duration and action not in {
            "archive",
            "display",
            "execute",
            "play",
            "read",
            "stream",
            "use",
        }:
            raise ODRLSerializationError(
                "A duration constraint cannot be mapped safely to "
                f"elapsedTime for action {action!r}."
            )
        for item in (count, duration, absolute_date, recipient, event):
            if item and item not in serialized:
                serialized.append(item)
        if not serialized:
            raise ODRLSerializationError(
                "Constraint has no lossless supported ODRL mapping: "
                f"{constraint_text!r}"
            )

    purpose_text = re.sub(
        r"^for\s+",
        "",
        _clean_text(purpose),
        flags=re.IGNORECASE,
    ).strip()
    purpose_text = re.sub(r"\s+only$", "", purpose_text, flags=re.I).strip()
    if purpose_text:
        serialized.append(
            {
                "leftOperand": "purpose",
                "operator": "eq",
                "rightOperand": _typed_value(purpose_text, "string"),
            }
        )

    for item in serialized:
        if item["leftOperand"] not in vocabulary["left_operands"]:
            raise ODRLSerializationError(
                f"Unknown ODRL left operand: {item['leftOperand']}"
            )
        if item["operator"] not in vocabulary["operators"]:
            raise ODRLSerializationError(
                f"Unknown ODRL operator: {item['operator']}"
            )
    return serialized


def _permission_duty_action(constraint: Any) -> str | None:
    """Return the action phrase for a clearly stated permission duty.

    The legacy extraction schema has only one free-text ``constraint`` field,
    so older extractors place permission duties there. Recognise only explicit
    duty language; arbitrary unsupported constraints must continue to fail
    closed in ``_serialize_constraints``.
    """
    text = _clean_text(constraint)
    match = re.search(
        r"\b(?:subject\s+to|conditional\s+upon)\s+"
        r"(?:the\s+)?(?:duty|obligation)\s+(?:of|to)\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    phrase = _clean_text(match.group(1)).rstrip(".,;:")
    return phrase or None


def _is_before_policy_usage(text: Any) -> bool:
    """Identify an explicit requirement to act before using the permission."""
    normalized = _clean_text(text).casefold()
    return bool(
        re.search(r"\bbeforehand\b", normalized)
        or re.search(
            r"\bbefore\s+(?:the\s+)?(?:permitted\s+)?"
            r"(?:action|use|sharing|distribution)\b",
            normalized,
        )
    )


def _failure_mentions_action(text: Any, action: str) -> bool:
    """Return whether text explicitly describes failure of a known duty."""
    normalized = _clean_text(text).casefold()
    if not re.search(r"\b(?:without|fail(?:ed|ing)?\s+to|not)\b", normalized):
        return False
    action_patterns = {
        "anonymize": r"\banonymi[sz](?:e|ed|es|ing|ation)\b",
    }
    pattern = action_patterns.get(action, rf"\b{re.escape(action)}\w*\b")
    return bool(re.search(pattern, normalized))


def _informed_party(source_rule: dict[str, Any], action: str) -> str:
    """Extract the explicit recipient of an ``inform`` consequence."""
    if action != "inform":
        return ""
    supplied = _clean_text(source_rule.get("informed_party"))
    if supplied:
        return supplied
    source_text = _clean_text(source_rule.get("source_text"))
    match = re.search(
        r"\b(?:notify|inform)\s+(?:the\s+)?(.+?)(?:[.,;]|$)",
        source_text,
        flags=re.IGNORECASE,
    )
    return _clean_text(match.group(1)) if match else ""


def _serialize_failure_duty(
    source_rule: dict[str, Any],
    *,
    uid: str,
    inherited_target: str | None,
    vocabulary: dict[str, frozenset[str]],
) -> dict[str, Any]:
    """Serialize a flat remedial rule as a nested consequence Duty."""
    action = _canonical_action(source_rule.get("action"), vocabulary)
    duty: dict[str, Any] = {
        "uid": uid,
        "action": action,
    }
    target = _as_iri(
        "asset",
        source_rule.get("asset"),
        required=False,
    ) or inherited_target
    if target:
        duty["target"] = target

    informed_party = _informed_party(source_rule, action)
    if action == "inform" and not informed_party:
        raise ODRLSerializationError(
            "An inform consequence requires an explicit informed party."
        )
    if informed_party:
        duty["informedParty"] = _as_iri(
            "party",
            informed_party,
            required=True,
        )
    return duty


def _serialize_permission_duty(
    duty_phrase: str,
    consequence_rule: dict[str, Any] | None,
    *,
    uid: str,
    inherited_target: str | None,
    vocabulary: dict[str, frozenset[str]],
) -> dict[str, Any]:
    """Serialize a precondition Duty embedded in a flat permission rule."""
    action = _canonical_action(duty_phrase, vocabulary)
    duty: dict[str, Any] = {
        "uid": uid,
        "action": action,
    }
    if inherited_target:
        duty["target"] = inherited_target
    if _is_before_policy_usage(duty_phrase):
        duty["constraint"] = [{
            "leftOperand": "event",
            "operator": "lt",
            "rightOperand": {"@id": "odrl:policyUsage"},
        }]
    if consequence_rule is not None:
        duty["consequence"] = [
            _serialize_failure_duty(
                consequence_rule,
                uid=f"{uid}:consequence:1",
                inherited_target=inherited_target,
                vocabulary=vocabulary,
            )
        ]
    return duty


def _permission_duty_plan(
    rules: list[dict[str, Any]],
    vocabulary: dict[str, frozenset[str]],
) -> tuple[dict[int, tuple[str, int | None]], set[int]]:
    """Match legacy permission-duty text with a following consequence rule."""
    plan: dict[int, tuple[str, int | None]] = {}
    consumed_consequences: set[int] = set()
    for permission_index, rule in enumerate(rules):
        if _clean_text(rule.get("rule_type")).casefold() != "permission":
            continue
        duty_phrase = _permission_duty_action(rule.get("constraint"))
        if not duty_phrase:
            continue
        duty_action = _canonical_action(duty_phrase, vocabulary)
        consequence_index: int | None = None
        for candidate_index, candidate in enumerate(rules):
            if (
                candidate_index == permission_index
                or candidate_index in consumed_consequences
            ):
                continue
            if (
                _clean_text(candidate.get("rule_type")).casefold()
                != "obligation"
            ):
                continue
            relation_text = " ".join((
                _clean_text(candidate.get("source_text")),
                _clean_text(candidate.get("purpose")),
            ))
            if not re.search(
                r"\b(?:if|remed(?:y|ial)|consequence)\b",
                relation_text,
                flags=re.IGNORECASE,
            ):
                continue
            permission_assignee = _clean_text(rule.get("assignee")).casefold()
            candidate_assignee = _clean_text(
                candidate.get("assignee")
            ).casefold()
            if (
                permission_assignee
                and candidate_assignee
                and permission_assignee != candidate_assignee
            ):
                continue
            failure_text = " ".join((
                _clean_text(candidate.get("constraint")),
                _clean_text(candidate.get("source_text")),
            ))
            if not _failure_mentions_action(failure_text, duty_action):
                continue
            # Validate the consequence action before consuming the top-level
            # rule, so an unsupported action still fails without semantic loss.
            _canonical_action(candidate.get("action"), vocabulary)
            consequence_index = candidate_index
            consumed_consequences.add(candidate_index)
            break
        plan[permission_index] = (duty_phrase, consequence_index)
    return plan, consumed_consequences


def _policy_uid(rules: list[dict[str, Any]], supplied_uid: str | None) -> str:
    if supplied_uid:
        if not _is_absolute_iri(supplied_uid):
            raise ODRLSerializationError("policy_uid must be an absolute IRI.")
        return supplied_uid
    canonical = json.dumps(
        rules,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"urn:lexlegal:policy:{hashlib.sha256(canonical.encode()).hexdigest()}"


def validate_serialized_odrl_structure(
    document: dict[str, Any],
    *,
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> None:
    """Validate the structural subset produced by this serializer.

        Its purpose is to ensure the serializer did not generate an obviously malformed document before it is placed into ODRLState or sent to the external ODRL Engine.

          It currently checks:
              - @context is the expected ODRL context.
              - @type is Policy or Set.
              - The policy has an absolute IRI uid.
              - At least one permission, prohibition, or obligation exists.
              - Each rule is an object.
              - Actions exist in the loaded ODRL vocabulary.
              - Permissions and prohibitions have a target IRI.
              - uid, target, assignee, and supported duty parties are absolute
                IRIs.
              - Constraints use known left operands and operators.
              - Every constraint has a rightOperand.
              - Permission duties and duty consequences are recursively valid.

      - validate_serialized_odrl_structure() checks the limited document format produced by our serializer.
            It is fast, local, and does not require the ODRL Engine service.
      - validate_odrl() performs authoritative, more comprehensive ODRL/RDF/SHACL validation through the external ODRL Engine.

      So it is not intended to replace ODRL Engine validation. It catches implementation and serialization problems early—for example, a missing target or unsupported action—while the engine checks broader standards compliance.

    """
    vocabulary = load_odrl_vocabulary(str(ontology_path))

    if document.get("@context") != ODRL_CONTEXT:
        raise ODRLSerializationError("ODRL document has an invalid @context.")
    if document.get("@type") not in {"Policy", "Set"}:
        raise ODRLSerializationError("ODRL document must be a Policy or Set.")

    uid = document.get("uid")

    if not isinstance(uid, str) or not _is_absolute_iri(uid):
        raise ODRLSerializationError("ODRL Policy requires one IRI uid.")

    def validate_rule(
        rule: Any,
        *,
        relationship: str,
        target_required: bool,
    ) -> None:
        if not isinstance(rule, dict):
            raise ODRLSerializationError("Every ODRL rule must be an object.")
        action = rule.get("action")
        if action not in vocabulary["actions"]:
            raise ODRLSerializationError(f"Unknown ODRL action: {action!r}")
        if target_required and not rule.get("target"):
            raise ODRLSerializationError(
                f"Every {relationship} requires one target IRI."
            )

        for iri_field in (
            "uid",
            "target",
            "assignee",
            "attributedParty",
            "informedParty",
        ):
            value = rule.get(iri_field)
            if value is not None and (
                not isinstance(value, str) or not _is_absolute_iri(value)
            ):
                raise ODRLSerializationError(
                    f"Rule {iri_field} must be an absolute IRI."
                )
        constraints = rule.get("constraint", [])
        if not isinstance(constraints, list):
            raise ODRLSerializationError("Rule constraint must be a list.")
        for constraint in constraints:
            if not isinstance(constraint, dict):
                raise ODRLSerializationError(
                    "Every ODRL constraint must be an object."
                )
            if constraint.get("leftOperand") not in vocabulary["left_operands"]:
                raise ODRLSerializationError("Unknown constraint left operand.")
            if constraint.get("operator") not in vocabulary["operators"]:
                raise ODRLSerializationError("Unknown constraint operator.")
            if "rightOperand" not in constraint:
                raise ODRLSerializationError(
                    "Constraint requires exactly one rightOperand."
                )

        if "duty" in rule:
            duties = rule["duty"]
            if relationship != "permission" or not isinstance(
                duties,
                list,
            ):
                raise ODRLSerializationError(
                    "Only a permission may contain a duty list."
                )
            for duty in duties:
                validate_rule(
                    duty,
                    relationship="duty",
                    target_required=False,
                )

        if "consequence" in rule:
            consequences = rule["consequence"]
            if relationship not in {"duty", "obligation"} or not isinstance(
                consequences,
                list,
            ):
                raise ODRLSerializationError(
                    "Only a permission duty or policy obligation may contain "
                    "a consequence list."
                )
            for consequence in consequences:
                validate_rule(
                    consequence,
                    relationship="consequence",
                    target_required=False,
                )

    rule_count = 0
    for property_name in RULE_TYPE_TO_PROPERTY.values():
        rules = document.get(property_name, [])
        if not isinstance(rules, list):
            raise ODRLSerializationError(f"{property_name} must be a list.")
        rule_count += len(rules)
        for rule in rules:
            validate_rule(
                rule,
                relationship=property_name,
                target_required=property_name in {"permission", "prohibition"},
            )
    if rule_count == 0:
        raise ODRLSerializationError(
            "ODRL Policy requires at least one permission, prohibition, or obligation."
        )


def serialize_odrl_document(
    validated_policy_rules: list[dict[str, Any]],
    *,
    policy_uid: str | None = None,
    ontology_path: str | Path = DEFAULT_ONTOLOGY_PATH,
) -> dict[str, Any]:
    """
        Convert atomic validated rules to a validated ODRL 2.2 JSON-LD Policy.
    """

    if not isinstance(validated_policy_rules, list) or not validated_policy_rules:
        raise ODRLSerializationError(
            "At least one validated policy rule is required for serialization."
        )
    if any(not isinstance(rule, dict) for rule in validated_policy_rules):
        raise ODRLSerializationError("Every validated policy rule must be an object.")

    vocabulary = load_odrl_vocabulary(str(ontology_path))
    duty_plan, consumed_consequences = _permission_duty_plan(
        validated_policy_rules,
        vocabulary,
    )
    # add an uid for each policy
    uid = _policy_uid(validated_policy_rules, policy_uid)

    document: dict[str, Any] = {
        "@context": ODRL_CONTEXT,
        "@type": "Policy",
        "uid": uid,
    }
    for index, source_rule in enumerate(validated_policy_rules, start=1):
        source_index = index - 1
        if source_index in consumed_consequences:
            continue
        rule_type = _clean_text(source_rule.get("rule_type")).casefold()
        property_name = RULE_TYPE_TO_PROPERTY.get(rule_type)
        if property_name is None:
            raise ODRLSerializationError(
                f"Unsupported rule_type for ODRL serialization: {rule_type!r}"
            )
        action = _canonical_action(source_rule.get("action"), vocabulary)
        target = _as_iri(
            "asset",
            source_rule.get("asset"),
            required=property_name in {"permission", "prohibition"},
        )
        assignee = _as_iri(
            "party",
            source_rule.get("assignee"),
            required=False,
        )
        identifier_seed = _clean_text(
            source_rule.get("policy_rule_id") or f"rule-{index}"
        )
        serialized_rule: dict[str, Any] = {
            "uid": f"{uid}:rule:{_slug(identifier_seed, str(index))}",
            "action": action,
        }
        if target:
            serialized_rule["target"] = target
        if assignee:
            serialized_rule["assignee"] = assignee
        attributed_party = _as_iri(
            "party",
            source_rule.get("attributed_party"),
            required=False,
        )
        if attributed_party:
            if action != "attribute":
                raise ODRLSerializationError(
                    "attributed_party is supported only for the attribute action."
                )
            serialized_rule["attributedParty"] = attributed_party
        duty_spec = duty_plan.get(source_index)
        constraints = _serialize_constraints(
            "" if duty_spec else source_rule.get("constraint"),
            source_rule.get("purpose"),
            action,
            vocabulary,
        )
        if constraints:
            serialized_rule["constraint"] = constraints
        if duty_spec:
            duty_phrase, consequence_index = duty_spec
            consequence_rule = (
                validated_policy_rules[consequence_index]
                if consequence_index is not None
                else None
            )
            serialized_rule["duty"] = [
                _serialize_permission_duty(
                    duty_phrase,
                    consequence_rule,
                    uid=f"{serialized_rule['uid']}:duty:1",
                    inherited_target=target,
                    vocabulary=vocabulary,
                )
            ]
        document.setdefault(property_name, []).append(serialized_rule)

    validate_serialized_odrl_structure(document, ontology_path=ontology_path)
    return document


# Compatibility name retained for tests and callers created before the local
# structural validator was renamed.
validate_odrl_document = validate_serialized_odrl_structure
