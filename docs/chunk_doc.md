
## Legal-document split rule

- Parent: one complete Article, recital group, or Annex.
- Article child: one numbered paragraph by default.
- Tiny plain paragraphs may be combined only when each is at most 80 estimated
  tokens, with at most three paragraphs and 250 estimated tokens per group.
- Definitions remain individual children.
- A paragraph is split at lettered points only when it exceeds 550 estimated
  tokens. Its governing introduction is repeated in every point-range child.
- Unnumbered fallback content targets 512 estimated tokens without cutting
  through a detected block.
- No arbitrary token overlap is added; legal headings and titles provide child
  context, and ``parent_id`` enables expansion to the complete legal unit.

## Token estimation

These uppercase module-level values are configuration constants. They are
defined once in `node_document_split.py` and are read by the relevant splitting
functions; they do not hold per-document state.

The splitter uses a lightweight estimate rather than a model-specific
tokenizer. Words and punctuation marks are counted separately. For example:

```text
The controller shall comply.
```

is estimated as five tokens: ``The``, ``controller``, ``shall``, ``comply``,
and ``.``. Consequently, these settings are stable grouping heuristics, not
exact limits for BGE or any other embedding model.

The size checks normally apply to the legal body being grouped. The final child
also repeats its Article or Annex heading and title, so its final estimated size
can be slightly higher than the applicable setting.

## Configuration reference

### `CHILD_TARGET_TOKENS = 512`

This is the preferred size for content that has no authoritative numbered legal
structure. It currently applies to:

- unnumbered Article fallback content;
- Annex paragraphs and sections.

It does **not** cause ordinary numbered Article paragraphs or definitions to be
combined until they reach 512 tokens.

The splitter adds complete blank-line-separated blocks in source order. When a
group reaches at least 512 estimated tokens, it emits that group as a child.

Example:

```text
Annex block 1: 220 tokens
Annex block 2: 210 tokens
Annex block 3: 100 tokens

Combined: 530 tokens -> emit one child containing blocks 1-3
```

No block is cut merely to land exactly on 512.

### `CHILD_HARD_MAX_TOKENS = 700`

This is the upper grouping safeguard for the same generic unnumbered and Annex
content. Before adding another complete block, the splitter checks whether the
candidate group would exceed 700 estimated tokens.

Example:

```text
Current group: 400 tokens
Next complete block: 350 tokens
Candidate group: 750 tokens

Result:
child 1 = the existing 400-token group
child 2 begins with the complete 350-token block
```

A single indivisible block may exceed 700 tokens. The splitter preserves that
block because cutting through arbitrary text would be less reliable than
temporarily exceeding the heuristic limit.

Together, the two generic settings mean:

```text
512 = preferred emission size
700 = do not add another complete block beyond this size
```

### `POINT_GROUP_TARGET_TOKENS = 400`

This is the preferred size for point-range children created from one oversized
numbered paragraph containing lettered points such as ``(a)``, ``(b)``, and
``(c)``.

The setting is used only after the complete paragraph has exceeded
`POINT_GROUP_MAX_TOKENS`. Once the current point range reaches approximately
400 estimated tokens, it is emitted instead of being packed unnecessarily close
to the maximum.

Example:

```text
Paragraph 2 governing introduction
points (a)-(f): approximately 415 tokens including the introduction
points (g)-(j): remaining points

Result:
child 1 = paragraph 2 introduction + points (a)-(f)
child 2 = paragraph 2 introduction + points (g)-(j)
```

Repeating the governing introduction ensures that the second child still makes
sense when retrieved independently.

### `POINT_GROUP_MAX_TOKENS = 550`

This value has two related roles:

1. A numbered paragraph containing lettered points remains intact when it is at
   or below 550 estimated tokens.
2. If the paragraph exceeds 550 tokens, the splitter divides it at lettered
   point boundaries and avoids adding another point when the candidate group
   would exceed 550 tokens.

For example, a 240-token paragraph containing points ``(a)-(f)`` remains one
child. This is the expected treatment of GDPR Article 6(1): the introductory
rule and all six legal bases stay together.

A 620-token paragraph containing points ``(a)-(j)`` is eligible for point
splitting. The resulting children target approximately 400 tokens and normally
remain at or below 550 tokens.

An individual point together with the repeated governing introduction can still
exceed 550 tokens. Such an indivisible legal point remains whole.

### `TINY_PARAGRAPH_MAX_TOKENS = 80`

This controls which ordinary numbered paragraphs are eligible for conservative
adjacent grouping. Each paragraph must contain at most 80 estimated tokens and
must not contain lettered points.

Example:

```text
paragraph 1: 33 tokens -> eligible
paragraph 2: 24 tokens -> eligible
paragraph 3: 35 tokens -> eligible
paragraph 4: 120 tokens -> not eligible; it remains standalone
```

A 90-token paragraph is not grouped merely because it is adjacent to a
20-token paragraph.

### `TINY_PARAGRAPH_GROUP_MAX_TOKENS = 250`

Eligible tiny paragraphs may be combined only while their complete group stays
at or below 250 estimated body tokens.

Example:

```text
paragraph 1: 70 tokens
paragraph 2: 75 tokens
paragraph 3: 65 tokens
combined: 210 tokens -> allowed
```

This produces a compact child rather than trying to reach the general 512-token
target. The repeated Article heading and title are added after this grouping
calculation.

### `TINY_PARAGRAPH_GROUP_MAX_BLOCKS = 3`

No more than three tiny numbered paragraphs may be placed in one child, even if
four or more would fit below 250 estimated tokens.

Example:

```text
paragraphs 1, 2, 3: 30 tokens each -> one child covering paragraphs 1-3
paragraph 4: 30 tokens             -> begins the next child
```

This prevents a large range of separate legal provisions from being hidden
inside one apparently small chunk.

All three tiny-paragraph conditions must be satisfied:

```text
each paragraph <= 80 tokens
combined group <= 250 tokens
number of paragraphs <= 3
```

### `RECITAL_GROUP_SIZE = 10`

Recitals use a different parent-child arrangement:

- each individual recital is a child retrieval unit;
- up to ten adjacent recitals are collected into one recital-group parent.

Example with 18 short recitals:

```text
parent 1 = recitals 1-10
  children = recital 1, recital 2, ..., recital 10

parent 2 = recitals 11-18
  children = recital 11, recital 12, ..., recital 18
```

The ten-recital limit is a maximum rather than a required size. Recital parent
groups are also emitted around 700 estimated tokens and are prevented from
accepting another recital above 1,000 estimated tokens. Therefore a group may
contain fewer than ten recitals when the recitals are long.

## How the settings interact

The settings belong to separate splitting paths:

| Content | Primary settings | Behaviour |
|---|---|---|
| Numbered Article paragraph | Tiny-paragraph settings | Standalone by default; only tiny plain neighbors may group |
| Oversized paragraph with lettered points | Point-group settings | Split at point boundaries with repeated introduction |
| Definition | None of the size-group settings | One definition per child |
| Unnumbered Article fallback | Child target and hard maximum | Group complete blank-line-separated blocks |
| Annex | Child target and hard maximum | Group complete blank-line-separated blocks |
| Recitals | Recital group size plus recital-specific token limits | Individual recital children under grouped parents |

The important distinction is that `target` means “emit when a useful size has
been reached,” while `max` means “do not add another complete unit past this
size.” Neither setting authorizes cutting through an indivisible legal unit.

Changing these constants can change child boundaries and therefore child IDs,
hashes, embeddings, and stored retrieval records. After changing them, the law
corpus should be split, embedded, and imported again rather than mixing old and
new chunk layouts.
