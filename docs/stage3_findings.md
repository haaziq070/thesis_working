# Stage 3 findings — supervised attack-identification classifier

## Headline result

Random Forest, 14 classes (13 CICIDS2017 attack types + benign), trained on
751,810 rows with a chronological per-source-file 70/30 split.

**Overall: 85% accuracy, weighted F1 = 0.86, macro F1 = 0.65.**

Full per-class report: `data/processed/stage3/classification_report_full.txt`
Confusion matrix (row-normalized heatmap): `data/processed/stage3/confusion_matrix.png`

## Leakage defenses applied (not optional cleanup — verified necessary)

1. **src_ip/dst_ip/src_port/dst_port excluded from features.** Verified
   directly against the raw data that CICIDS2017's fixed attacker IP
   (172.16.0.1) never appears in any of 50,000 sampled benign rows, while
   every attack type uses that same fixed IP. Including IP would let the
   model hit ~100% by keying on attacker identity rather than flow behavior.

2. **Chronological per-file split, not random row split.** Floods (DoS Hulk,
   port scans, DDoS) produce tens of thousands of near-identical consecutive
   rows; a random split would leak near-duplicates of the same burst across
   train/test. Each file's first 70% of timestamps -> train, last 30% ->
   test.

## Near-perfect classes: investigated, not waved through

Two classes scored near-perfect and were checked individually rather than
accepted at face value:

**DDoS_LOIT (F1=0.994, n=28,720 test rows).** Confusion matrix: 28,698/28,720
correct, only 22 misclassified as Benign — no suspicious clean-boundary
pattern. Top feature importances are dominated by inter-packet-arrival-time
statistics (fwd_packets_IAT_mean, packet_IAT_total, etc.), which is a
genuine, well-documented DDoS flood signature (near-zero, highly uniform
packet timing) — not a residual identity leak, since IP/port were already
excluded. Verdict: legitimate easy class.

**Port_Scan (F1=0.97 overall, but recall=0.999 on n=48,397 — missed by the
F1-based near-perfect check since precision=0.944 pulled the F1 down; caught
by a second row-normalized-recall pass over the confusion matrix).**
Breakdown: of 48,397 true port-scan flows, 48,335 correctly identified (62
errors, mostly to Benign). But 1,813 Benign flows and 1,071 Botnet_ARES flows
were misclassified *as* Port_Scan (the source of the precision hit). Story:
port-scan traffic (many short connection attempts) has a highly consistent,
hard-to-miss flow signature (drives recall up), while some other short/
probe-like traffic (background benign connections, botnet C2 beaconing)
superficially resembles it in flow-statistic space (drives Port_Scan's
precision down). Coherent, non-leakage explanation.

**Lesson for future stages:** an F1-based near-perfect check alone is not
enough — it can miss a class whose recall is suspiciously perfect if
precision happens to be lower. Check precision and recall separately, not
just the combined F1, when auditing for leakage.

## Honest weak spots (reported, not hidden)

- **Web_XSS: F1=0.30.** Web-layer attacks are hard to fingerprint from flow
  statistics alone (no HTTP payload visibility in this feature set) —
  expected, not a bug.
- **Web_Brute_Force: F1=0.39.** Same story as XSS.
- **Web_SQL_Injection: F1=0.857 but only 8 test examples.** Flagged
  automatically as statistically untrustworthy — do not present as a solid
  number in the thesis without this caveat.
- **Heartbleed: only 4 test examples.** Essentially noise; report but do not
  interpret.
- **Benign precision = 0.49 despite recall = 0.91.** A real number of attack
  flows get misclassified as benign (false negatives from a security
  standpoint). This is a genuine limitation to carry into Stage 6, not
  something to paper over — it means, as currently trained, a meaningful
  fraction of real attacks would slip through unflagged.

## What this means for Stage 4

Stage 4's DQN correlator will consume this classifier's *predicted*
attack_type as part of its state representation, per the two-stage design.
Given Benign's low precision and XSS/Brute-Force's weak F1, the correlator's
input labels will sometimes be wrong — this is realistic (a real SOC's
identifier is never perfect either) and should be treated as a feature of
the evaluation, not hidden. Stage 5's honest evaluation should include a
brief sensitivity discussion: how much does Stage 3's imperfection propagate
into Stage 4/5's correlation quality.
