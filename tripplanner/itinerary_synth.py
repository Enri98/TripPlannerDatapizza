"""Deterministic day-by-day itinerary synthesis from validated specialist outputs."""

from __future__ import annotations

from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, Field

from tripplanner.contracts import StandardAgentResult, TripSpec


RainRisk = Literal["low", "high"]
OutputLanguage = Literal["en", "it"]


class ItineraryActivity(BaseModel):
    name: str
    period: Literal["morning", "afternoon", "evening"]
    indoor: bool


class DayPlan(BaseModel):
    day_index: int = Field(ge=1)
    date: str
    destination: str
    weather_risk: RainRisk
    weather_note: str
    activities: list[ItineraryActivity] = Field(default_factory=list)
    alternatives: list[str] = Field(default_factory=list)
    transport_notes: list[str] = Field(default_factory=list)


class ItineraryPlan(BaseModel):
    language: OutputLanguage
    title: str
    days: list[DayPlan] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ItinerarySynthesizer:
    """Build a day-by-day itinerary using only provided validated inputs."""

    def synthesize(
        self,
        *,
        tripspec: TripSpec,
        results: dict[str, StandardAgentResult],
        language_override: OutputLanguage | None = None,
    ) -> ItineraryPlan:
        language = self._resolve_language(tripspec=tripspec, language_override=language_override)
        days: list[DayPlan] = []
        warnings: list[str] = []
        day_index = 1

        for leg_index, leg in enumerate(tripspec.legs):
            weather = results.get(f"weather_leg_{leg_index}")
            poi = results.get(f"poi_leg_{leg_index}")
            transport = results.get(f"transport_leg_{leg_index - 1}_{leg_index}") if leg_index > 0 else None

            weather_by_date = self._weather_by_date(weather)
            poi_items = list((poi.data if poi else {}).get("pois", []))

            current = leg.date_range.start_date
            while current <= leg.date_range.end_date:
                date_key = current.isoformat()
                risk = self._weather_risk(weather_by_date.get(date_key))
                weather_note = _text(
                    language,
                    "weather_rain" if risk == "high" else "weather_clear",
                    risk=risk,
                )
                activities = self._build_activities(
                    language=language,
                    pois=poi_items,
                    rainy=(risk == "high"),
                )
                alternatives = self._build_alternatives(language=language, rainy=(risk == "high"))
                transport_notes = self._transport_notes(language=language, transport=transport, is_leg_start=(current == leg.date_range.start_date))

                if not weather_by_date:
                    warnings.append(_text(language, "missing_weather_warning"))
                if not poi_items:
                    warnings.append(_text(language, "missing_poi_warning"))

                days.append(
                    DayPlan(
                        day_index=day_index,
                        date=date_key,
                        destination=(leg.geo.place_name if leg.geo else leg.destination_text),
                        weather_risk=risk,
                        weather_note=weather_note,
                        activities=activities,
                        alternatives=alternatives,
                        transport_notes=transport_notes,
                    )
                )
                day_index += 1
                current += timedelta(days=1)

        deduped_warnings = list(dict.fromkeys(warnings))
        return ItineraryPlan(
            language=language,
            title=_text(language, "title"),
            days=days,
            warnings=deduped_warnings,
        )

    def _resolve_language(
        self,
        *,
        tripspec: TripSpec,
        language_override: OutputLanguage | None,
    ) -> OutputLanguage:
        if language_override in {"en", "it"}:
            return language_override
        return "it" if tripspec.request_context.output_language.lower().startswith("it") else "en"

    def _weather_by_date(
        self,
        result: StandardAgentResult | None,
    ) -> dict[str, dict]:
        if result is None:
            return {}
        rows = result.data.get("daily", [])
        if not isinstance(rows, list):
            return {}
        output: dict[str, dict] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            date = row.get("date")
            if isinstance(date, str) and date:
                output[date] = row
        return output

    def _weather_risk(self, row: dict | None) -> RainRisk:
        if not row:
            return "low"
        precip = row.get("precipitation_probability_max")
        code = row.get("weather_code")
        rainy_codes = {51, 53, 55, 56, 57, 61, 63, 65, 80, 81, 82, 95}
        try:
            if precip is not None and float(precip) >= 60:
                return "high"
        except (TypeError, ValueError):
            pass
        try:
            if code is not None and int(code) in rainy_codes:
                return "high"
        except (TypeError, ValueError):
            pass
        return "low"

    def _build_activities(
        self,
        *,
        language: OutputLanguage,
        pois: list[dict],
        rainy: bool,
    ) -> list[ItineraryActivity]:
        indoor: list[str] = []
        outdoor: list[str] = []
        for poi in pois:
            if not isinstance(poi, dict):
                continue
            name = str(poi.get("name") or "").strip()
            if not name:
                continue
            tags = poi.get("tags", {})
            if self._is_indoor(tags):
                indoor.append(name)
            else:
                outdoor.append(name)

        selected: list[str] = []
        if rainy:
            selected.extend(indoor[:2])
            if len(selected) < 2:
                selected.extend(outdoor[: 2 - len(selected)])
        else:
            selected.extend(outdoor[:2])
            if len(selected) < 2:
                selected.extend(indoor[: 2 - len(selected)])

        while len(selected) < 2:
            selected.append(_text(language, "fallback_activity"))

        periods: list[Literal["morning", "afternoon", "evening"]] = ["morning", "afternoon"]
        activities: list[ItineraryActivity] = []
        for idx, name in enumerate(selected[:2]):
            activities.append(
                ItineraryActivity(
                    name=name,
                    period=periods[idx],
                    indoor=name in indoor or name == _text(language, "fallback_activity"),
                )
            )
        return activities

    def _build_alternatives(self, *, language: OutputLanguage, rainy: bool) -> list[str]:
        if rainy:
            return [_text(language, "rain_alternative")]
        return []

    def _transport_notes(
        self,
        *,
        language: OutputLanguage,
        transport: StandardAgentResult | None,
        is_leg_start: bool,
    ) -> list[str]:
        if not is_leg_start or transport is None:
            return []

        options = transport.data.get("options", [])
        if not isinstance(options, list) or not options:
            return [_text(language, "transport_generic")]

        notes: list[str] = []
        for option in options[:2]:
            title = option.get("title") if isinstance(option, dict) else None
            if not title:
                continue
            notes.append(_text(language, "transport_option", title=str(title)))
        return notes or [_text(language, "transport_generic")]

    def _is_indoor(self, tags: object) -> bool:
        if not isinstance(tags, dict):
            return False
        tourism = str(tags.get("tourism", "")).lower()
        amenity = str(tags.get("amenity", "")).lower()
        indoor_tag = str(tags.get("indoor", "")).lower()
        if indoor_tag in {"yes", "true", "1"}:
            return True
        if tourism in {"museum", "gallery"}:
            return True
        return amenity in {"theatre", "cinema", "library"}


def _text(language: OutputLanguage, key: str, **kwargs: str) -> str:
    catalog = {
        "en": {
            "title": "Day-by-day itinerary",
            "weather_rain": "High rain risk. Prefer indoor activities.",
            "weather_clear": "Weather looks manageable for outdoor plans.",
            "fallback_activity": "Local indoor discovery walk",
            "rain_alternative": "Indoor backup: museum or covered market.",
            "transport_option": "Transfer option: {title}",
            "transport_generic": "Use a practical transfer option and verify timing locally.",
            "missing_weather_warning": "Weather data missing for some days.",
            "missing_poi_warning": "POI data missing for some days.",
        },
        "it": {
            "title": "Itinerario giorno per giorno",
            "weather_rain": "Rischio pioggia alto. Preferisci attivita al chiuso.",
            "weather_clear": "Meteo gestibile per attivita all'aperto.",
            "fallback_activity": "Passeggiata di scoperta al chiuso",
            "rain_alternative": "Alternativa al chiuso: museo o mercato coperto.",
            "transport_option": "Opzione trasferimento: {title}",
            "transport_generic": "Usa un trasferimento pratico e verifica gli orari sul posto.",
            "missing_weather_warning": "Dati meteo mancanti per alcuni giorni.",
            "missing_poi_warning": "Dati POI mancanti per alcuni giorni.",
        },
    }
    template = catalog[language][key]
    return template.format(**kwargs)
