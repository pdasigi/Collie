from collie.extractor_utils import ConstraintExtractor, raise_exception
from collie.constraints import *
import random

random.seed(20934)

LENGTH_CONSTRAINTS = {
    "num_words_equals": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("==")],
        },
        target_range=list(range(50, 110, 10)) + list(range(100, 1000, 50))
    ),
    "num_words_atleast_50-200": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation(">=")],
        },
        target_range=list(range(50, 250, 50)),
        # We want the text to not be much longer than the specified limit.
        post_extract=lambda x: x if x < 220 else raise_exception()
    ),
    "num_words_atleast_250-400": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation(">=")],
        },
        target_range=list(range(250, 450, 50)),
        # We want the text to not be much longer than the specified limit.
        post_extract=lambda x: x if x < 420 else raise_exception()
    ),
    "num_words_atleast_450-600": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation(">=")],
        },
        target_range=list(range(450, 650, 50)),
        # We want the text to not be much longer than the specified limit.
        post_extract=lambda x: x if x < 620 else raise_exception()
    ),
    "num_words_atmost_250-400": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("<=")],
        },
        target_range=list(range(250, 450, 50)),
        # We want the text to not be much shorter than the specified limit.
        post_extract=lambda x: x if x > 220 else raise_exception()
    ),
    "num_words_atmost_450-600": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("<=")],
        },
        target_range=list(range(450, 650, 50)),
        # We want the text to not be much shorter than the specified limit.
        post_extract=lambda x: x if x > 420 else raise_exception()
    ),
    "num_words_atmost_650-800": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("<=")],
        },
        target_range=list(range(650, 850, 50)),
        # We want the text to not be much shorter than the specified limit.
        post_extract=lambda x: x if x > 620 else raise_exception()
    ),
    "num_chars_equals": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("character")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("==")],
        },
        target_range=list(range(200, 5000, 50))
    ),
    "num_chars_atleast_500-2000": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("character")],
        "transformation": [ForEach(Count(), first_only=True)],
        "relation": [Relation(">=")],
        },
        target_range=list(range(500, 2500, 500)),
        # We want the text to not be much longer than the specified limit.
        post_extract=lambda x: x if x < 2200 else raise_exception()
    ),
    "num_chars_atleast_2500-4000": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("character")],
        "transformation": [ForEach(Count(), first_only=True)],
        "relation": [Relation(">=")],
        },
        target_range=list(range(2500, 4500, 500)),
        # We want the text to not be much longer than the specified limit.
        post_extract=lambda x: x if x < 4100 else raise_exception()
    ),
    "num_chars_atmost_200-1000": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("character")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("<=")],
        },
        target_range=list(range(200, 1000, 100)),
        # We want the text to not be much shorter than the specified limit.
        post_extract=lambda x: x if x > 100 else raise_exception()
    ),
    "num_chars_atmost_1100-2000": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("character")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("<=")],
        },
        target_range=list(range(1100, 2100, 100)),
        # We want the text to not be much shorter than the specified limit.
        post_extract=lambda x: x if x > 1000 else raise_exception()
    ),
    "num_sents_equals": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("sentence")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("==")],
        },
        target_range=list(range(5, 30))
    ),
    "num_sents_atleast": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("sentence")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation(">=")],
        },
        target_range=list(range(2, 20))
    ),
    "num_sents_atmost": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("sentence")],
            "transformation": [ForEach(Count(), first_only=True)],
            "relation": [Relation("<=")],
        },
        target_range=list(range(5, 10))
    ),
    "num_words_in_sents_atmost": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("sentence")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count())],
            "relation": [Relation("<=")],
            "reduction": [Reduction("all")],
        },
        target_range=list(range(15, 25, 5)),
        # Only extract passages with at least three sentences and all sentences are long enough
        post_extract=lambda x: x if len(x) > 2 and min(x) > 5 else raise_exception()
    ),
    "num_words_in_sents_atleast": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("sentence")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(Count())],
            "relation": [Relation(">=")],
            "reduction": [Reduction("all")],
        },
        target_range=list(range(5, 20, 5)),
        # Only extract passages with at least three sentences and all sentences are not too long
        post_extract=lambda x: x if len(x) > 2 and max(x) < 20 else raise_exception()

    ),
    "num_chars_in_sents_atmost": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("sentence")],
            "target_level": [TargetLevel("character")],
            "transformation": [ForEach(Count())],
            "relation": [Relation("<=")],
            "reduction": [Reduction("all")],
        },
        target_range=list(range(100, 300, 50)),
        # Only extract passages with at least three sentences
        post_extract=lambda x: x if len(x) > 2 else raise_exception()
    ),
    "num_chars_in_sents_atleast": ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("sentence")],
            "target_level": [TargetLevel("character")],
            "transformation": [ForEach(Count())],
            "relation": [Relation(">=")],
            "reduction": [Reduction("all")],
        },
        target_range=list(range(50, 200, 50)),
        # Only extract passages with at least three sentences
        post_extract=lambda x: x if len(x) > 2 else raise_exception()
    ),
}

POSITION_CONSTRAINTS = {
    "first_word": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("word")],
        "transformation": [ForEach(Position([0]), first_only=True)],
        "relation": [Relation("==")],
        },
        target_range=None,
        # Do not extract passages where the target "words" are numbers or punctuation.
        post_extract=lambda x: x if all([re.fullmatch("[a-z]+", xi.lower()) for xi in x]) else raise_exception()
    ),
    "last_word": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("word")],
        "transformation": [ForEach(Position([-1]), first_only=True)],
        "relation": [Relation("==")],
        },
        target_range=None,
        # Do not extract passages where the target "words" are numbers or punctuation.
        post_extract=lambda x: x if all([re.fullmatch("[a-z]+", xi.lower()) for xi in x]) else raise_exception()
    ),
    "first_and_second_word": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("word")],
        "transformation": [ForEach(Position([0, 1]), first_only=True)],
        "relation": [Relation("==")],
        },
        target_range=None,
        # Do not extract passages where the target "words" are numbers or punctuation.
        post_extract=lambda x: x if all([re.fullmatch("[a-z]+", xi.lower()) for xi in x]) else raise_exception()
    ),
    "first_and_last_word": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("word")],
        "transformation": [ForEach(Position([0, -1]), first_only=True)],
        "relation": [Relation("==")],
        },
        target_range=None,
        # Do not extract passages where the target "words" are numbers or punctuation.
        post_extract=lambda x: x if all([re.fullmatch("[a-z]+", xi.lower()) for xi in x]) else raise_exception()
    ),
    "first_sentence": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("sentence")],
        "transformation": [ForEach(Position([0]), first_only=True)],
        "relation": [Relation("==")],
        },
        target_range=None,
        # Only extract passages with at least two sentences
        #post_extract=lambda x: x if len(x) > 1 else raise_exception()
    ),
    "last_sentence": ConstraintExtractor(
    init_range={
        "input_level": [InputLevel("passage")],
        "target_level": [TargetLevel("sentence")],
        "transformation": [ForEach(Position([-1]), first_only=True)],
        "relation": [Relation("==")],
        },
        target_range=None,
        # Only extract passages with at least two sentences
        #post_extract=lambda x: x if len(x) > 1 else raise_exception()
    ),
}

LENGTH_AND_POSITION_CONSTRAINTS = {
    "num_sents_atmost_last_word": [
        ConstraintExtractor(
            init_range={
                "input_level": [InputLevel("passage")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Count(), first_only=True)],
                "relation": [Relation("<=")],
            },
            target_range=list(range(5, 10))
        ),
        ConstraintExtractor(
            init_range={
                "input_level": [InputLevel("passage")],
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(Position([-1]), first_only=True)],
                "relation": [Relation("==")],
                },
                target_range=None,
                # Do not extract passages where the target "words" are numbers or punctuation.
                post_extract=lambda x: x if all([re.fullmatch("[a-z]+", xi.lower()) for xi in x]) else raise_exception()
        ),
    ],
    "num_sents_atleast_first_word": [
        ConstraintExtractor(
            init_range={
                "input_level": [InputLevel("passage")],
                "target_level": [TargetLevel("sentence")],
                "transformation": [ForEach(Count(), first_only=True)],
                "relation": [Relation(">=")],
            },
            target_range=list(range(2, 20))
        ),
        ConstraintExtractor(
            init_range={
                "input_level": [InputLevel("passage")],
                "target_level": [TargetLevel("word")],
                "transformation": [ForEach(Position([0]), first_only=True)],
                "relation": [Relation("==")],
                },
                target_range=None,
                # Do not extract passages where the target "words" are numbers or punctuation.
                post_extract=lambda x: x if all([re.fullmatch("[a-z]+", xi.lower()) for xi in x]) else raise_exception()
        ),
    ],
}


def get_word_presence_constraints(word_frequencies, min_freq=10, max_freq=1000, sample_size=100):
    infrequent_words = []
    for word, freq in word_frequencies.items():
        if freq < min_freq or freq > max_freq:
            continue
        if re.fullmatch("[a-z]+", word):
            infrequent_words.append(word)
    sample = random.sample(infrequent_words, sample_size)
    word_presence_constraint = ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(..., first_only=True)],
            "relation": [Relation("in")],
            },
            target_range=sample
    )
    word_absence_constraint = ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("word")],
            "transformation": [ForEach(..., first_only=True)],
            "relation": [Relation("not in")],
            },
            target_range=sample
    )
    char_absence_constraint = ConstraintExtractor(
        init_range={
            "input_level": [InputLevel("passage")],
            "target_level": [TargetLevel("character")],
            "transformation": [ForEach(..., first_only=True)],
            "relation": [Relation("not in")],
            },
            target_range=[",", "!", "?"]
    )
    presence_constraints = {
        "word_presence": word_presence_constraint,
        "word_absence": word_absence_constraint,
        "char_absence": char_absence_constraint,
        "word_presence_and_word_presence": [word_presence_constraint, word_presence_constraint],
        "word_presence_and_word_absence": [word_presence_constraint, word_absence_constraint],
    }
    for name, constraint in LENGTH_CONSTRAINTS.items():
        presence_constraints[f"word_presence_and_{name}"] = [
            constraint,
            word_presence_constraint,
        ]

    return presence_constraints

