"""Sentence generation constraints"""
import random
from collie.constraints import *
from collie.extractor_utils import ConstraintExtractor, raise_exception


# sentence count constraint
SENT_CONSTRAINTS = {
    "one_sent_with_length_equals": 
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            target_range = list(range(10,20))
        ),
    "one_short_sent": 
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Count()],
                "relation": [Relation("<=")]
            },
            target_range = list(range(2,5))
        ),
    "one_long_sent": 
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Count()],
                "relation": [Relation(">=")]
            },
            target_range = list(range(20,30))
        ),
    "words_in_positions_1_2":
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Position([1, 2])],
                "relation": [Relation("==")],
            },
            target_range = None
        ),
    "words_in_positions_2_4":
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Position([2, 4])],
                "relation": [Relation("==")],
            },
            target_range = None
        ),
}

COMPLEX_SENT_CONTRAINTS = {
    "c05": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            target_range = list(range(10,20))
        ),
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Position([3, 7, 10])],
                "relation": [Relation("==")],
            },
            target_range = None
        )
    ],
    "one_sent_with_short_words":
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("word")],
                "target_level": [TargetLevel("character")],
                "transformation": [ForEach(Count())],
                "relation": [Relation("<=")],
                "reduction": [Reduction("all")]
            },
            # this will pull out all examples greater than 10 and reject all others
            post_extract=lambda x: max(x) if max(x) > 10 else raise_exception() 
        ),
    "one_sent_with_long_words":
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("word")],
                "target_level": [TargetLevel("character")],
                "transformation": [ForEach(Count())],
                "relation": [Relation(">=")],
                "reduction": [Reduction("all")]
            },
            # this will pull out all examples less than 10 and reject all others
            post_extract=lambda x: max(x) if max(x) < 11 else raise_exception() 
        ),
    "c06a": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [Count()],
                "relation": [Relation(">=")]
            },
            post_extract=lambda x: x if x > 7 else raise_exception()
            # post_extract = lambda x: breakpoint()
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("word")],
                "target_level": [TargetLevel("character")],
                "transformation": [ForEach(Count())],
                "relation": [Relation("<=")],
                "reduction": [Reduction("all")]
            },
            # this will pull out all examples less than 7 and reject all others
            post_extract=lambda x: max(x) if max(x) < 8 else raise_exception() 
            # post_extract=lambda x: breakpoint()
        )
    ],
}
