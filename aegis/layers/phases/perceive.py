"""PERCEIVE phase — read the world and update affect.

Extracted verbatim from ``Substrate._perceive`` (spec §3.9). The substrate had
grown to 1763 lines with seven more systems still to land; each phase now lives
where it can be read and tested on its own. Behaviour is unchanged — the bodies
were moved, not rewritten.
"""
import logging

from aegis.event_bus import Event, Layer

from aegis.layers.phases.context import TickContext

logger = logging.getLogger("aegis.substrate")


async def run(substrate, ctx: TickContext) -> None:
    substrate.cycle_phase = "perceive"
    perception = substrate.world.perceive()

    # Sensor cortex
    sensor_data = substrate.sensors.read_all()
    perception["sensors"] = sensor_data

    # Emotional perception — reward from REAL system metrics
    reward = substrate._compute_reward()
    context = {
        "tick": substrate.tick_count,
        "new_knowledge": substrate._tick_new_concepts > 0,
        "error": substrate.health.consecutive_errors > 0,
        "unexpected": substrate.health.consecutive_errors > 3,
        "repetitive": substrate.emotions.mood_duration > 15,
    }
    substrate.emotions.update(reward, context)

    # Update consciousness based on emotion
    substrate.consciousness.update_mode(substrate.emotions.mood, substrate.emotions.energy, substrate.emotions.arousal)

    # Archetype activation
    update_archetypes(substrate)

    substrate.memory.add_working({"phase": "perceive", "data": perception, "mood": substrate.emotions.mood})
    await substrate.event_bus.publish(Event(
        source=Layer.SUBSTRATE, target=Layer.MEMORY,
        event_type="perception", payload=perception
    ))

def update_archetypes(substrate) -> None:
    """Pick the archetype whose activation condition the current affect
    satisfies, and log what it would do."""
    for arch in substrate.archetypes_list:
        if arch.should_activate(substrate.emotions.mood, substrate.emotions.energy):
            substrate.active_archetype = arch
            break
    else:
        if substrate.archetypes_list:
            substrate.active_archetype = substrate.archetypes_list[0]

    if substrate.active_archetype:
        action_desc = substrate.active_archetype.act(
            substrate.consciousness.mode,
            substrate.goals.get_current_focus().get("name", "idle") if substrate.goals.get_current_focus() else "idle"
        )
        substrate.active_archetype.log_experience(
            substrate.tick_count, substrate.emotions.mood,
            substrate.emotions.success_rate, action_desc
        )

    if substrate.tick_count % 10 == 0:
        substrate.geopolitics.update_influence()
