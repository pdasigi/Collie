"""Paragraph generation constraints"""
import random
from collie.constraints import *
from collie.extractor_utils import ConstraintExtractor, raise_exception, no_sents_filter


# sentence count constraint
PASSAGE_CONSTRAINTS = {
    "min_sentences_per_paragraph":
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("paragraph")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Count())],
                "relation": [Relation(">=")],
                "reduction": [Reduction("all")]
            },
            target_range=[2, 3, 4, 5]
        ),
    "num_sentences_per_paragraph":
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("paragraph")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Count())],
                "relation": [Relation("==")],
                "reduction": [Reduction("all")]
            },
            target_range=list(range(2, 11))
        ),
    "min_num_characters":
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("passage")],
                "target_level": [TargetLevel("character")],
                "transformation": [Count()],
                "relation": [Relation(">=")],
            },
            target_range=list(range(500, 1050, 50))
        ),
    "max_num_characters":
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("passage")],
                "target_level": [TargetLevel("character")],
                "transformation": [Count()],
                "relation": [Relation("<=")],
            },
            target_range=list(range(150, 550, 50)),
            post_extract=lambda x: x if len(x) > 100 else raise_exception()
        ),
    "without_word":
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(...)],
                "relation": [Relation("not in")],
                "reduction": [Reduction("all")]
            },
            target_range=["the", "are", "of", "in", "for", "and"]
        ),
    "with_word":
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(...)],
                "relation": [Relation("in")],
            },
            target_range=["strange", "plain", "definitely", "numerous", "rigorous", "moist", "pressure", "magic"]
        ),
    "without_character":
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("character")],
                "transformation": [ForEach(...)],
                "relation": [Relation("not in")],
                "reduction": [Reduction("all")]
            },
            target_range=[",", "!", "?", ";"]
        ),
    "with_character":
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("character")],
                "transformation": [ForEach(...)],
                "relation": [Relation("in")],
            },
            target_range=["!", "?", "<", ">", "(", ")", ";"]
        ),
}

COMPLEX_PASSAGE_CONSTRAINTS = {
    "c13": [
        ConstraintExtractor(
            init_range = {
                "target_level": [TargetLevel("paragraph")],
                "transformation": [Count()],
                "relation": [Relation("==")]
            },
            post_extract=lambda x: x if x in range(3,5) else raise_exception() 
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("paragraph")],
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(Count())],
                "relation": [Relation("==")],
            },
        ),
    ],
    "c14": [
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("paragraph")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Position(-1))],
                "relation": [Relation("==")],
                "reduction": [Reduction("all")]
            },
            # make sure there are between 2 and 5 paragraphs, and each sentence target is not too long
            post_extract=lambda x: x if len(x) in range(2, 5) else raise_exception() #and all([len(y.split(" ")) < 20 for y in x]) else raise_exception()
        ),
        ConstraintExtractor(
            init_range = {
                "input_level": [InputLevel("paragraph")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Count())],
                "relation": [Relation(">=")],
                "reduction": [Reduction("all")]
            },
            target_range=[2]
        ),
    ]
}