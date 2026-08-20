"""
Unified event schema shared by every data source (DARPA2000 now; CICIDS2017,
UNSW-NB15, DVWA later). Every Stage-2 parser must emit a dataframe with
exactly these columns, so that Stage 3/4/5 code never has to know which
source an event came from.

Columns
-------
event_id          str   unique within source_dataset
timestamp         datetime64[ns], UTC-naive but consistently parsed
src_ip, dst_ip    str
src_port, dst_port  float (NaN where not applicable, e.g. ICMP)
service           str   raw protocol/service tag from the source (e.g.
                        "telnet", "sunrpc/u", "eco/i" for DARPA; will be
                        HTTP method/URL pattern for web-tier sources)
tier              str   one of "network", "web", "host"
vantage           str   sensor location, e.g. "inside", "dmz" (DARPA-specific;
                        "N/A" for sources without multiple vantage points)
source_dataset    str   e.g. "darpa2000_lldos1.0", "cicids2017" — used to
                        namespace campaign_id and to enforce the DARPA-as-
                        pure-holdout rule
campaign_id       str   ground-truth campaign this event belongs to, OR
                        "NONE" if it is a distractor / non-campaign event
                        that happens to share the time window
kill_chain_stage  str   one of KILL_CHAIN_STAGES below, or "unknown"
attack_type       str   coarse label; for DARPA this is a network-attack
                        vocabulary (recon/exploit/ddos), NOT the Stage-3
                        SQLi/XSS/scan/benign taxonomy — the two are kept
                        deliberately separate, see note below
is_campaign_link  bool  True if this event is part of a real escalating
                        attack chain, False if it is recon/traffic that
                        never escalated (a genuine hard negative, not a
                        synthetic one)

IMPORTANT: DARPA2000's attack_type vocabulary (recon/exploit/ddos) is
NOT the same taxonomy the Stage-3 classifier is trained to predict
(SQLi/XSS/scan/benign) — DARPA predates HTTP-layer attacks entirely, it is
network/host only. DARPA's attack_type here is derived directly from the
dataset's own phase structure (ground truth), not from a classifier
prediction, and is used only for descriptive labeling — it is never fed into
Stage 3 training. Keeping this separate is one of the scope commitments from
the proposal: attack *identification* and *correlation* are different
problems, and DARPA only ever participates in the correlation (Stage 4/5)
side, as the held-out evaluation set.
"""

UNIFIED_COLUMNS = [
    "event_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "service",
    "tier",
    "vantage",
    "source_dataset",
    "campaign_id",
    "kill_chain_stage",
    "attack_type",
    "is_campaign_link",
]

KILL_CHAIN_STAGES = [
    "recon",       # host/service discovery, sweeps, probes
    "exploit",     # initial compromise
    "install",     # backdoor/trojan/zombie installation
    "action",      # actions on objective (e.g. DDoS launch, exfil)
    "unknown",
]

# DARPA2000 LLDOS phase -> kill-chain-stage mapping. This is the documented
# structure of the LLDOS 1.0 / 2.0.2 scenarios as designed by MIT Lincoln
# Laboratory for the 2000 DARPA intrusion-detection evaluation (the scenario
# is: IP sweep -> sadmind RPC probe -> sadmind buffer-overflow exploit ->
# install mstream DDoS trojan -> launch DDoS). This mapping is read off the
# dataset's own phase split (phase-N-dump / phase-N.list files), not invented
# by us — cite the Lincoln Lab LLDOS scenario documentation in the thesis.
DARPA_PHASE_TO_STAGE = {
    1: "recon",     # IP sweep
    2: "recon",     # sadmind RPC probe (vulnerability scan)
    3: "exploit",   # sadmind buffer-overflow break-in
    4: "install",   # install DDoS (mstream) trojan on compromised hosts
    5: "action",    # launch the DDoS
}

DARPA_PHASE_TO_ATTACK_TYPE = {
    1: "recon",
    2: "recon",
    3: "exploit",
    4: "backdoor_install",
    5: "ddos",
}

# CICIDS2017 (BCCC re-extraction) label -> kill-chain-stage mapping. CICIDS2017
# does not ship official multi-stage campaign labels -- these are individual
# attack-type files, each a discrete red-team exercise. The stage mapping
# below is a standard MITRE ATT&CK-style categorization (Brute Force / Port
# Scan -> reconnaissance & credential access; the actual exploitation
# techniques -> exploit; DoS/DDoS/botnet impact traffic -> action-on-
# objective), not something read off explicit dataset ground truth the way
# DARPA's phase structure is. Document this distinction in the thesis: DARPA's
# kill_chain_stage is dataset-native ground truth, CICIDS2017's is a derived
# label.
CICIDS_LABEL_TO_STAGE = {
    "FTP-Patator": "recon",
    "SSH-Patator": "recon",
    "Web_Brute_Force": "recon",
    "Port_Scan": "recon",
    "Web_XSS": "exploit",
    "Web_SQL_Injection": "exploit",
    "Heartbleed": "exploit",
    "DoS Hulk": "action",
    "DoS GoldenEye": "action",
    "DoS Slowhttptest": "action",
    "DoS Slowloris": "action",
    "DDoS": "action",
    "Botnet_ARES": "action",
    "Benign": "unknown",
}

# Which day-of-week each raw CSV belongs to, and whether it's the day's
# benign baseline or an attack-type file. Read directly off this BCCC release
# (confirmed by inspecting each file's own timestamp range) and cross-checked
# against the officially published CIC-IDS2017 attack schedule
# (Sharafaldin et al. 2018) -- Tue=brute force, Wed=DoS, Thu=web attacks,
# Fri=botnet/portscan/DDoS. Cite that schedule in the thesis when justifying
# the day groupings used for campaign construction.
CICIDS_FILE_TO_DAY = {
    "ftp_patator.csv": "tue",
    "ssh_patator-new.csv": "tue",
    "tuesday_benign.csv": "tue",
    "dos_slowloris.csv": "wed",
    "dos_slowhttptest.csv": "wed",
    "dos_hulk.csv": "wed",
    "dos_golden_eye.csv": "wed",
    "heartbleed.csv": "wed",
    "wednesday_benign.csv": "wed",
    "web_brute_force.csv": "thu",
    "web_xss.csv": "thu",
    "web_sql_injection.csv": "thu",
    "thursday_benign.csv": "thu",
    "botnet_ares.csv": "fri",
    "portscan.csv": "fri",
    "ddos_loit.csv": "fri",
    "friday_benign.csv": "fri",
    "monday_benign.csv": "mon",
}

# --- Stage 3: supervised attack-identification classifier ---
#
# Columns to EXCLUDE from the feature matrix. This is not a routine cleanup
# step -- src_ip/dst_ip/src_port/dst_port are a genuine, verified data-leakage
# vector in this dataset: every single attack type (SQLi, XSS, port scan, DoS,
# botnet, everything) was generated by the SAME fixed attacker VM
# (172.16.0.1) hitting the SAME fixed victim (192.168.10.50), while benign
# traffic involves thousands of distinct real-world IPs and NEVER includes
# 172.16.0.1 (verified directly against the raw data: 0 occurrences across
# 50,000 sampled benign rows). Leaving IP/port in the feature set would let
# the classifier hit ~100% accuracy by learning "is the attacker's fixed IP
# present" as a perfect proxy for "is this an attack" -- a shortcut that has
# nothing to do with genuine flow-based attack-signature recognition, and
# exactly the kind of leakage this project is committed to catching before it
# inflates a result.
CICIDS_LEAKAGE_COLUMNS = [
    "flow_id", "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "label",
]

# The literal 4-class taxonomy named in the thesis proposal. Reported as a
# secondary "headline" view alongside the full extended multi-class model
# (see docs/stage3_design_note.md for why both are reported).
PROPOSAL_HEADLINE_CLASSES = ["Web_SQL_Injection", "Web_XSS", "Port_Scan", "Benign"]
