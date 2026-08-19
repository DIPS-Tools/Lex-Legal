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
    "anonymize": "anonymize",
    "copy": "reproduce",
    "delete": "delete",
    "deleted": "delete",
    "download": "reproduce",
    "downloaded": "reproduce",
    "erase": "delete",
    "execute": "execute",
    "export": "export",
    "modify": "modify",
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
        "left_operands": {"count", "elapsedTime", "purpose"},
        "operators": {"eq", "lteq", "gteq"},
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
        for item in (count, duration):
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
              - uid, target, and assignee are absolute IRIs.
              - Constraints use known left operands and operators.
              - Every constraint has a rightOperand.

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

    rule_count = 0
    for property_name in RULE_TYPE_TO_PROPERTY.values():
        rules = document.get(property_name, [])
        if not isinstance(rules, list):
            raise ODRLSerializationError(f"{property_name} must be a list.")
        rule_count += len(rules)
        for rule in rules:
            if not isinstance(rule, dict):
                raise ODRLSerializationError("Every ODRL rule must be an object.")
            action = rule.get("action")
            if action not in vocabulary["actions"]:
                raise ODRLSerializationError(f"Unknown ODRL action: {action!r}")
            if property_name in {"permission", "prohibition"} and not rule.get("target"):
                raise ODRLSerializationError(
                    f"Every {property_name} requires one target IRI."
                )

            for iri_field in ("uid", "target", "assignee"):
                value = rule.get(iri_field)
                if value is not None and (
                    not isinstance(value, str) or not _is_absolute_iri(value)
                ):
                    raise ODRLSerializationError(
                        f"Rule {iri_field} must be an absolute IRI."
                    )
            for constraint in rule.get("constraint", []):
                if constraint.get("leftOperand") not in vocabulary["left_operands"]:
                    raise ODRLSerializationError("Unknown constraint left operand.")
                if constraint.get("operator") not in vocabulary["operators"]:
                    raise ODRLSerializationError("Unknown constraint operator.")
                if "rightOperand" not in constraint:
                    raise ODRLSerializationError(
                        "Constraint requires exactly one rightOperand."
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
    # add an uid for each policy
    uid = _policy_uid(validated_policy_rules, policy_uid)

    document: dict[str, Any] = {
        "@context": ODRL_CONTEXT,
        "@type": "Policy",
        "uid": uid,
    }
    for index, source_rule in enumerate(validated_policy_rules, start=1):
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
        constraints = _serialize_constraints(
            source_rule.get("constraint"),
            source_rule.get("purpose"),
            action,
            vocabulary,
        )
        if constraints:
            serialized_rule["constraint"] = constraints
        document.setdefault(property_name, []).append(serialized_rule)

    validate_serialized_odrl_structure(document, ontology_path=ontology_path)
    return document
