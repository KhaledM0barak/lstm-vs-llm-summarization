"""Prompt variants for the LLM baseline.

The assignment requires at least two prompt variants so that the comparison is
not an artifact of one weak prompt. The two here differ along the axis that
matters most for ROUGE against CNN/DailyMail: whether the prompt tells the model
what the *reference* summaries look like.

  variant A ("plain")      - a natural, unconditioned summarization request.
                             This is what a user would write without having seen
                             the dataset.
  variant B ("style-matched") - additionally specifies the length, sentence
                             count, and register of CNN/DailyMail highlights.

Variant A measures "how good is an LLM at summarizing"; variant B measures "how
good is an LLM at this dataset's reference style". Reporting both separates
genuine capability from metric-fitting, which is the point of the exercise.
"""

from __future__ import annotations

from dataclasses import dataclass

SYSTEM_PLAIN = "You are a helpful assistant that writes concise news summaries."

SYSTEM_STYLE = (
    "You write summaries in the style of CNN/DailyMail article highlights: "
    "3 to 4 short, declarative, self-contained sentences totalling roughly 55 "
    "words. Each sentence states one concrete fact from the article — a name, a "
    "number, a place, an action. You never editorialize, never add background "
    "the article does not contain, and never write an introductory clause such "
    "as 'This article discusses'."
)

USER_PLAIN = "Summarize the following news article.\n\nArticle:\n{article}\n\nSummary:"

USER_STYLE = (
    "Write the highlights for the following news article. Output only the "
    "highlight sentences as a single paragraph, with no bullet points, no "
    "labels, and no preamble.\n\nArticle:\n{article}\n\nHighlights:"
)


@dataclass(frozen=True)
class PromptVariant:
    key: str
    name: str
    system: str
    user_template: str
    description: str

    def render_user(self, article: str) -> str:
        return self.user_template.format(article=article)


VARIANTS: dict[str, PromptVariant] = {
    "A": PromptVariant(
        key="A",
        name="plain",
        system=SYSTEM_PLAIN,
        user_template=USER_PLAIN,
        description="Natural summarization request with no dataset-specific styling.",
    ),
    "B": PromptVariant(
        key="B",
        name="style-matched",
        system=SYSTEM_STYLE,
        user_template=USER_STYLE,
        description="Specifies CNN/DailyMail highlight length, sentence count, and register.",
    ),
}


def build_messages(
    variant: PromptVariant,
    article: str,
    exemplars: list[dict] | None = None,
) -> list[dict]:
    """Build the message list for one request.

    Few-shot exemplars are passed as real prior conversation turns rather than
    being pasted into a single user message: that is the shape the API is
    designed around, and it keeps the exemplar boundaries unambiguous.
    """
    messages: list[dict] = []
    for ex in exemplars or []:
        messages.append({"role": "user", "content": variant.render_user(ex["article"])})
        messages.append({"role": "assistant", "content": ex["summary"]})
    messages.append({"role": "user", "content": variant.render_user(article)})
    return messages
