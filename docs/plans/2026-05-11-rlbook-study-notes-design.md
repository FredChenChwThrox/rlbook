# RLBook Study Notes Design

**Goal:** Convert the repository from a chapter-by-chapter Chinese translation style into a study-notes style repository while preserving the existing chapter structure, key formulas, pseudocode, and major learning flow.

## Scope

- Keep chapter numbering and most section headings.
- Compress long explanatory translation prose into short note-style summaries.
- Preserve important formulas and pseudocode where they help learning.
- Add plain-language explanations after important formulas so readers can understand what each formula does.
- Reposition the repository as a personal study-notes companion rather than a full translation.

## Content Rules

### Chapter structure

Each chapter should continue to look like a chapter, not a one-page abstract. Existing sections should be retained where practical.

### Paragraph compression

Long translated paragraphs should be rewritten into:

- concise summary sentences
- bullet-point takeaways
- short intuitive explanations
- reading tips

### Formula handling

Formulas should be kept when they are central to the topic. Each important formula should be followed by a short explanation covering:

- what the formula measures or computes
- what its main symbols mean
- why it matters in the algorithm or concept

### Pseudocode handling

Pseudocode should be kept when it is educationally useful. Surrounding explanation should be shortened to note-style descriptions.

### Copyright-risk reduction

The repository should no longer describe itself as a translation-first project. Wording should emphasize:

- personal study notes
- explanatory adaptation
- upstream reference
- non-official status

## File Strategy

- Update `README.md` to describe the repository as a study-notes edition.
- Keep `NOTICE.md` and `COPYRIGHT.md` aligned with the study-notes positioning.
- Rewrite all chapter files and the appendix in place.

## Validation

Success means:

- the repository reads as notes rather than continuous translation
- formulas remain understandable
- chapter structure is still recognizable
- upstream and copyright notices remain visible
