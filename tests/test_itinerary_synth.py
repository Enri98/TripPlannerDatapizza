"""Tests for deterministic itinerary synthesis and EN/IT rendering."""

from __future__ import annotations

from datetime import datetime, timezone

from tripplanner.contracts import StandardAgentResult, TripSpec
from tripplanner.itinerary_synth import ItinerarySynthesizer


def _tripspec(*, output_language: str = "en", second_leg: bool = False) -> TripSpec:
    legs = [
        {
            "destination_text": "Rome",
            "geo": {
                "lat": 41.9,
                "lon": 12.4,
                "bbox": [41.8, 12.3, 42.0, 12.7],
                "place_name": "Rome",
                "country_code": "it",
            },
            "date_range": {"start_date": "2026-03-10", "end_date": "2026-03-11"},
        }
    ]
    if second_leg:
        legs.append(
            {
                "destination_text": "Florence",
                "geo": {
                    "lat": 43.8,
                    "lon": 11.3,
                    "bbox": [43.7, 11.2, 43.9, 11.5],
                    "place_name": "Florence",
                    "country_code": "it",
                },
                "date_range": {"start_date": "2026-03-12", "end_date": "2026-03-12"},
            }
        )

    return TripSpec.model_validate(
        {
            "request_context": {
                "now_ts": "2026-02-20T12:00:00Z",
                "timezone": "Europe/Rome",
                "input_language": "en",
                "output_language": output_language,
            },
            "budget": {"amount": 1200, "currency": "EUR", "scope": "total"},
            "legs": legs,
            "preferences": {"tags": ["culture", "food"]},
            "constraints": {"pace": "standard", "mobility": "public_transport"},
        }
    )


def _result(data: dict, cache_key: str = "cache:key") -> StandardAgentResult:
    return StandardAgentResult.model_validate(
        {
            "data": data,
            "evidence": [
                {
                    "source": "test",
                    "title": "mock",
                    "snippet": "mock",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                }
            ],
            "confidence": 0.9,
            "warnings": [],
            "cache_key": cache_key,
        }
    )


def test_synthesizer_builds_day_by_day_with_weather_aware_alternative() -> None:
    tripspec = _tripspec(output_language="en")
    results = {
        "weather_leg_0": _result(
            {
                "daily": [
                    {"date": "2026-03-10", "precipitation_probability_max": 80, "weather_code": 63},
                    {"date": "2026-03-11", "precipitation_probability_max": 10, "weather_code": 1},
                ]
            }
        ),
        "poi_leg_0": _result(
            {
                "pois": [
                    {"name": "Capitoline Museums", "tags": {"tourism": "museum"}},
                    {"name": "Villa Borghese", "tags": {"leisure": "park"}},
                ]
            }
        ),
    }

    plan = ItinerarySynthesizer().synthesize(tripspec=tripspec, results=results)

    assert plan.language == "en"
    assert plan.title == "Day-by-day itinerary"
    assert [day.date for day in plan.days] == ["2026-03-10", "2026-03-11"]
    assert plan.days[0].weather_risk == "high"
    assert plan.days[0].activities[0].name == "Capitoline Museums"
    assert plan.days[0].alternatives == ["Indoor backup: museum or covered market."]
    assert plan.days[1].weather_risk == "low"
    assert plan.days[1].activities[0].name == "Villa Borghese"


def test_synthesizer_uses_italian_when_output_language_is_it() -> None:
    tripspec = _tripspec(output_language="it")
    results = {
        "weather_leg_0": _result(
            {"daily": [{"date": "2026-03-10", "precipitation_probability_max": 75, "weather_code": 63}]}
        ),
        "poi_leg_0": _result({"pois": []}),
    }

    plan = ItinerarySynthesizer().synthesize(tripspec=tripspec, results=results)

    assert plan.language == "it"
    assert plan.title == "Itinerario giorno per giorno"
    assert plan.days[0].weather_note == "Rischio pioggia alto. Preferisci attivita al chiuso."
    assert "Dati POI mancanti per alcuni giorni." in plan.warnings


def test_synthesizer_supports_forced_language_override_and_transport_notes() -> None:
    tripspec = _tripspec(output_language="en", second_leg=True)
    results = {
        "weather_leg_0": _result({"daily": [{"date": "2026-03-10"}]}),
        "poi_leg_0": _result({"pois": [{"name": "Pantheon", "tags": {"tourism": "attraction"}}]}),
        "weather_leg_1": _result({"daily": [{"date": "2026-03-12"}]}),
        "poi_leg_1": _result({"pois": [{"name": "Uffizi Gallery", "tags": {"tourism": "museum"}}]}),
        "transport_leg_0_1": _result(
            {
                "options": [
                    {"title": "Fast train Rome to Florence"},
                    {"title": "Regional train alternative"},
                ]
            }
        ),
    }

    plan = ItinerarySynthesizer().synthesize(
        tripspec=tripspec,
        results=results,
        language_override="it",
    )

    assert plan.language == "it"
    assert plan.days[2].destination == "Florence"
    assert plan.days[2].transport_notes == [
        "Opzione trasferimento: Fast train Rome to Florence",
        "Opzione trasferimento: Regional train alternative",
    ]
