## Changes

## Troubles (optional)

## Checklist for self-check
- [ ] The author is marked as an assigne and assigned mandatory reviewers.
- [ ] Required labels marked
- [ ] Specified related tasks and/or related PRs.
- [ ] Specified Changes.
- [ ] All unspecified fields in the PR description deleted.
- [ ] New code covered by unit-tests or doesn't need this

## Checklist for reviewers
Pull request:
- CI passed successfully _(with a green check mark)_.
- PR is atomic, by volume no more than 400 (+-) corrected lines.

Design:
- System design corresponds to the agreements on structure and architecture on the project.
- The code is decomposed into necessary and sufficient components.

Complexity:
- The code is clear, easy to read, functions are small, no more than 50 lines.
- The logic is not overcomplicated, there is no overengineering (no code sections that may be needed in the future, but no one knows about it).

Tests:
- Updated or added tests for mandatory components.
- The tests are correct, helpful, and well designed/developed.

Naming:
- The naming of variables, methods, classes and other components is understandable.

Comments:
- The comments are understandable and helpful and consistent with [google pydocstyle](https://github.com/NilsJPWerner/autoDocstring/blob/HEAD/docs/google.md).

Documentation:
- All labels are correct
- Technical documentation updated (after approval, updates last reviewer).
