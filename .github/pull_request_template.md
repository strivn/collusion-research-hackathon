<!--
Title (typed in the field above, not here): one change, imperative, stands alone.
"Delete the FizzBuzz RPC", not "Deleted...", not "Fix bug", not "Phase 1".
If it needs an "and", it is two PRs.

Rules for this description:
- First sentence of every section is the whole answer. Lines after it are optional depth.
- Plain clauses. No em dashes. No semicolon fragment stacking. No drama, no cleverness.
- Say each thing once. If two sections would repeat, cut one.
- Scale to the diff: a 3-line change gets a 3-line description. In a big PR, one
  sentence per specific thing, as bullet points.
- Every claim carries its evidence: a number, a test name, a log line.
- One name per concept, held constant.
-->

## Requirement / Issue (Why)
<!-- What you want, one sentence. Then at most a line or two on why it matters now. -->

## Solution (What)
<!-- The behavior change, one sentence. Then the details a user would notice. -->

## Implementation (How)
<!-- Numbered trace: start from the user action, walk the code path to the result.
     Name code as module.function so each step is findable. Tag each step (new) or
     (existing); a (new) step says where it lives, e.g. (new, actions.py).
     Fold small failure handling into the steps. -->
1.

## Verify
<!-- Test count and gates in one line. Then the strongest evidence: device results,
     measurements, mutation checks. -->

**Red first**
<!-- The failing run at the test( commit, trimmed to the summary lines. Delete if the
     PR has no test( commit. -->

## Caveats
<!-- Known limits and first guesses. Delete the section if none. -->

<!-- Sections below only when real. Delete otherwise.

## Exception handling
What can go wrong and what the code does about it, when too big to fold into How.

## Breaking changes
Migration or reinstall required, and how.

## Notes for reviewer
Where to look hardest, what you are unsure of.

## Out of scope
What this deliberately skips and where it is handled instead.

## Files
- `path/to/file` and one plain line each, when the diff spans enough files that a map helps.

## Stack
PR N of M. Depends on #X. Enables #Y.
-->
