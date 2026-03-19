import math
import random

from instaoptima.instruction import Instruction


OBJECTIVE_FIELDS = ("performance", "length", "perplexity")


def dominates(candidate: Instruction, competitor: Instruction) -> bool:
    candidate_values = candidate.objectives.as_tuple()
    competitor_values = competitor.objectives.as_tuple()
    return (
        all(candidate_value <= competitor_value for candidate_value, competitor_value in zip(candidate_values, competitor_values))
        and any(candidate_value < competitor_value for candidate_value, competitor_value in zip(candidate_values, competitor_values))
    )


def non_dominated_sort(population: list[Instruction]) -> list[list[Instruction]]:
    domination_counts: dict[int, int] = {}
    dominated_sets: dict[int, list[Instruction]] = {}
    fronts: list[list[Instruction]] = [[]]

    for individual in population:
        domination_counts[id(individual)] = 0
        dominated_sets[id(individual)] = []
        for other in population:
            if individual is other:
                continue
            if dominates(individual, other):
                dominated_sets[id(individual)].append(other)
            elif dominates(other, individual):
                domination_counts[id(individual)] += 1
        if domination_counts[id(individual)] == 0:
            individual.rank = 0
            fronts[0].append(individual)

    front_index = 0
    while front_index < len(fronts) and fronts[front_index]:
        next_front: list[Instruction] = []
        for individual in fronts[front_index]:
            for dominated_individual in dominated_sets[id(individual)]:
                domination_counts[id(dominated_individual)] -= 1
                if domination_counts[id(dominated_individual)] == 0:
                    dominated_individual.rank = front_index + 1
                    next_front.append(dominated_individual)
        if next_front:
            fronts.append(next_front)
        front_index += 1

    return fronts


def assign_crowding_distance(front: list[Instruction]) -> None:
    if not front:
        return

    for individual in front:
        individual.crowding_distance = 0.0

    if len(front) <= 2:
        for individual in front:
            individual.crowding_distance = float("inf")
        return

    for objective_field in OBJECTIVE_FIELDS:
        front.sort(key=lambda instruction: getattr(instruction.objectives, objective_field))
        front[0].crowding_distance = float("inf")
        front[-1].crowding_distance = float("inf")

        min_value = getattr(front[0].objectives, objective_field)
        max_value = getattr(front[-1].objectives, objective_field)
        scale = max(max_value - min_value, 1e-12)

        for index in range(1, len(front) - 1):
            if math.isinf(front[index].crowding_distance):
                continue
            previous_value = getattr(front[index - 1].objectives, objective_field)
            next_value = getattr(front[index + 1].objectives, objective_field)
            front[index].crowding_distance += (next_value - previous_value) / scale


def select_next_population(
    population: list[Instruction],
    population_size: int,
    random_replacement_ratio: float = 0.0,
) -> tuple[list[Instruction], list[Instruction]]:
    fronts = non_dominated_sort(population)
    selected: list[Instruction] = []
    pareto_front: list[Instruction] = fronts[0] if fronts else []

    for front in fronts:
        assign_crowding_distance(front)
        if len(selected) + len(front) <= population_size:
            selected.extend(front)
            continue

        sorted_front = sorted(front, key=lambda item: item.crowding_distance, reverse=True)
        remaining_slots = population_size - len(selected)
        selected.extend(sorted_front[:remaining_slots])
        break

    if random_replacement_ratio > 0 and len(selected) > 1:
        selected = _apply_random_replacement(
            selected=selected,
            source_population=population,
            random_replacement_ratio=random_replacement_ratio,
        )

    return selected, pareto_front


def _apply_random_replacement(
    selected: list[Instruction],
    source_population: list[Instruction],
    random_replacement_ratio: float,
) -> list[Instruction]:
    replacement_count = min(
        max(1, int(len(selected) * random_replacement_ratio)),
        len(selected) - 1,
    )
    elite = selected[0]
    candidates = [individual for individual in source_population if individual is not elite]
    if not candidates:
        return selected

    survivors = selected[:-replacement_count]
    replacements = random.sample(candidates, k=min(replacement_count, len(candidates)))
    return survivors + replacements


def pareto_front(population: list[Instruction]) -> list[Instruction]:
    fronts = non_dominated_sort(population)
    return fronts[0] if fronts else []
