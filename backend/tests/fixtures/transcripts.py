"""Transcript fixtures.

Realistic word-level transcripts are hard to check in as opaque blobs, so they
are generated here from readable text: each word is given a duration
proportional to its length, with a longer gap after sentence-ending
punctuation. That reproduces the timing structure the segmenter and the
boundary snapper actually rely on.
"""

from __future__ import annotations

from app.providers.base import TranscriptChunk, TranscriptionResult, TranscriptWord

#: A monologue with deliberate structure: a weak opening, a strong hook with a
#: clean payoff, a mid-sentence dangling reference, and a rambling tail.
SAMPLE_MONOLOGUE = """\
Okay so welcome back to the channel, uh, today we are going to talk about \
something I get asked about constantly. Before we start, please subscribe.
Most people completely misunderstand how compounding works. They think it is \
about patience. It is not. It is about survival.
The reason is simple. Compounding only works if you never interrupt it, and \
almost everyone interrupts it. One bad year of panic selling erases nine good \
years of returns, and the maths on that is brutal.
So what do you actually do? You automate the boring part and you make the \
decision once, not every month.
And that is why it works so well for people who never look at their accounts.
Anyway, um, that is basically all I wanted to say about that, you know, and I \
guess we can move on to the next topic which is kind of related.
"""

HINGLISH_MONOLOGUE = """\
Dekho, sabse bada mistake jo log karte hain wo ye hai ki wo apni income pe \
dhyan nahi dete. Sirf saving pe focus karte hain.
Asal mein aapki saving rate limited hai. Income unlimited hai. Isliye das \
ghante raise maangne mein lagao, das saal coupon katne se behtar hai.
"""


def build_transcript(
    text: str,
    *,
    language: str = "en",
    words_per_second: float = 2.6,
    sentence_pause: float = 0.55,
    paragraph_pause: float = 1.15,
    start_offset: float = 0.0,
) -> TranscriptionResult:
    """Turn readable text into a word-timed :class:`TranscriptionResult`."""
    chunks: list[TranscriptChunk] = []
    cursor = start_offset

    for paragraph in [p.strip() for p in text.strip().split("\n") if p.strip()]:
        words: list[TranscriptWord] = []
        chunk_start = cursor
        for token in paragraph.split():
            # Longer words take longer to say; this keeps rates plausible.
            duration = max(0.12, len(token) / (words_per_second * 4.2))
            words.append(
                TranscriptWord(
                    word=token,
                    start=round(cursor, 3),
                    end=round(cursor + duration, 3),
                    probability=0.94,
                )
            )
            cursor += duration
            if token.endswith((".", "!", "?")):
                cursor += sentence_pause
            else:
                cursor += 0.06
        chunks.append(
            TranscriptChunk(
                start=round(chunk_start, 3),
                end=round(words[-1].end, 3),
                text=paragraph,
                words=words,
            )
        )
        cursor += paragraph_pause

    return TranscriptionResult(
        language=language,
        language_probability=0.99,
        duration=round(cursor, 3),
        chunks=chunks,
        provider="fixture",
        model="fixture",
        metadata={"synthetic": False, "source": "tests.fixtures.transcripts"},
    )


def sample_transcript() -> TranscriptionResult:
    return build_transcript(SAMPLE_MONOLOGUE)


def hinglish_transcript() -> TranscriptionResult:
    return build_transcript(HINGLISH_MONOLOGUE, language="hi")
