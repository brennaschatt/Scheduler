from shiny import App, reactive, render, ui
import pandas as pd
from ortools.sat.python import cp_model
try:
    import anthropic as _anthropic_mod
except ImportError:
    _anthropic_mod = None

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ROLE_HIERARCHY = {
    "Manager":     ["Manager", "Lead Server", "Server", "Host"],
    "Lead Server": ["Lead Server", "Server", "Host"],
    "Server":      ["Server"],
    "Host":        ["Host"],
}

ROLES           = ["Manager", "Lead Server", "Server", "Host"]
ALL_DAYS        = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
DAYS            = ALL_DAYS  # kept for backwards compat; filtered at runtime
SHIFT_TYPES     = ["AM", "PM"]
HOURS_PER_SHIFT = 6
DAY_ORDER       = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MAX_EMPLOYEES   = 30

# All possible shift IDs (across all 7 days)
ALL_SHIFTS = [f"{d}_{s}" for d in ALL_DAYS for s in SHIFT_TYPES]

# ── Default example schedule (shown on first load) ───────────────────────────
# Pre-built valid schedule using the sample employee dataset.
# Replaced the moment the manager clicks Generate Schedule.
_DEFAULT_SCHED_ROWS = [
    {"Shift":"Mon_AM","Day":"Mon","Shift Type":"AM","# Staff":6,"Workers":"Maggie (Manager), Steve (Lead Server), Brenda (Server), Bo (Server), Hector (Server), John (Host)"},
    {"Shift":"Mon_PM","Day":"Mon","Shift Type":"PM","# Staff":6,"Workers":"Morgan (Manager), Sarah (Lead Server), Billy (Server), Bailey (Server), Holly (Server), Jane (Host)"},
    {"Shift":"Tue_AM","Day":"Tue","Shift Type":"AM","# Staff":6,"Workers":"Maggie (Manager), Brian (Lead Server), Brenda (Server), Bailey (Server), Hector (Server), Kevin (Host)"},
    {"Shift":"Tue_PM","Day":"Tue","Shift Type":"PM","# Staff":6,"Workers":"Sandra (Manager), Steve (Lead Server), Billy (Server), Bo (Server), Holly (Server), Kim (Host)"},
    {"Shift":"Wed_AM","Day":"Wed","Shift Type":"AM","# Staff":6,"Workers":"Morgan (Manager), Sarah (Lead Server), Brenda (Server), Bo (Server), Holly (Server), Leo (Host)"},
    {"Shift":"Wed_PM","Day":"Wed","Shift Type":"PM","# Staff":6,"Workers":"Sandra (Manager), Brian (Lead Server), Billy (Server), Bailey (Server), Hector (Server), Lily (Host)"},
    {"Shift":"Thu_AM","Day":"Thu","Shift Type":"AM","# Staff":6,"Workers":"Sandra (Manager), Steve (Lead Server), Brenda (Server), Bailey (Server), Holly (Server), John (Host)"},
    {"Shift":"Thu_PM","Day":"Thu","Shift Type":"PM","# Staff":6,"Workers":"Maggie (Manager), Brian (Lead Server), Billy (Server), Bo (Server), Hector (Server), Jane (Host)"},
    {"Shift":"Fri_AM","Day":"Fri","Shift Type":"AM","# Staff":6,"Workers":"Morgan (Manager), Sarah (Lead Server), Brenda (Server), Bo (Server), Holly (Server), Kevin (Host)"},
    {"Shift":"Fri_PM","Day":"Fri","Shift Type":"PM","# Staff":6,"Workers":"Sandra (Manager), Brian (Lead Server), Billy (Server), Bailey (Server), Hector (Server), Kim (Host)"},
    {"Shift":"Sat_AM","Day":"Sat","Shift Type":"AM","# Staff":6,"Workers":"Maggie (Manager), Steve (Lead Server), Brenda (Server), Bailey (Server), Hector (Server), Leo (Host)"},
    {"Shift":"Sat_PM","Day":"Sat","Shift Type":"PM","# Staff":6,"Workers":"Morgan (Manager), Sarah (Lead Server), Billy (Server), Bo (Server), Holly (Server), Lily (Host)"},
]

_ROLES = {
    "Maggie":"Manager","Morgan":"Manager","Sandra":"Manager",
    "Steve":"Lead Server","Sarah":"Lead Server","Brian":"Lead Server",
    "Brenda":"Server","Billy":"Server","Bo":"Server","Bailey":"Server","Hector":"Server","Holly":"Server",
    "John":"Host","Jane":"Host","Kevin":"Host","Kim":"Host","Leo":"Host","Lily":"Host",
}

def _build_default_summary():
    from collections import defaultdict
    shifts_by_emp = defaultdict(list)
    for row in _DEFAULT_SCHED_ROWS:
        sid = row["Shift"]
        st  = row["Shift Type"]
        for w in row["Workers"].split(", "):
            name = w[:w.rfind(" (")].strip()
            shifts_by_emp[name].append((sid, st))
    rows = []
    for name, assigned in shifts_by_emp.items():
        n    = len(assigned)
        rows.append({
            "Name":            name,
            "Role":            _ROLES.get(name, ""),
            "Shifts":          n,
            "Hours":           n * 6,
            "AM Shifts":       sum(1 for _,st in assigned if st=="AM"),
            "PM Shifts":       sum(1 for _,st in assigned if st=="PM"),
            "Pref %":          "N/A",
            "Assigned Shifts": ", ".join(sid for sid,_ in assigned),
        })
    return pd.DataFrame(rows)

DEFAULT_SCHED_DF = pd.DataFrame(_DEFAULT_SCHED_ROWS)
DEFAULT_SUMM_DF  = _build_default_summary()
DEFAULT_METRICS  = {
    "Shifts Scheduled":         12,
    "Total Staff Slots":        72,
    "Avg Hours / Employee":     24.0,
    "Max Hours (any emp)":      36,
    "Min Hours (any emp)":      12,
    "Hours Std Dev (Fairness)": "—",
    "Pref Satisfaction":        "N/A (example)",
    "Staffing Coverage":        "100%",
}


# ─────────────────────────────────────────────
#  DATA HELPERS
# ─────────────────────────────────────────────

def clean_emp(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    renames = {}
    for col in df.columns:
        low = col.lower()
        if low in ("name", "employee", "employee_name") and col != "Name":
            renames[col] = "Name"
        elif low in ("role", "position") and col != "Role":
            renames[col] = "Role"
        elif low in ("max_hours", "maxhours", "max hours") and col != "Max_Hours":
            renames[col] = "Max_Hours"
    df = df.rename(columns=renames)
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    df["Max_Hours"] = pd.to_numeric(df["Max_Hours"], errors="coerce").fillna(40).astype(int)
    return df


def clean_shift(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    for col in ["Total_Staff", "Manager", "Lead_Server", "Server", "Host"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


def smart_read(file_info):
    if file_info is None:
        return None
    path = file_info[0]["datapath"]
    name = file_info[0]["name"].lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(path, on_bad_lines="skip", engine="python")
        else:
            return pd.read_excel(path)
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1", on_bad_lines="skip", engine="python")
        except Exception:
            return None


# ─────────────────────────────────────────────
#  TEMPLATE → DATAFRAMES
# ─────────────────────────────────────────────

def build_emp_df_from_inputs(input, n_emp, open_days):
    active_shifts = [f"{d}_{s}" for d in open_days for s in SHIFT_TYPES]
    rows = []
    for i in range(n_emp):
        name = input[f"t_name_{i}"]()
        if not name or not name.strip():
            continue
        row = {
            "Name":      name.strip(),
            "Role":      input[f"t_role_{i}"](),
            "Max_Hours": int(input[f"t_maxh_{i}"]()),
        }
        for sid in active_shifts:
            raw_avail = input[f"t_avail_{i}_{sid}"]()
            row[f"{sid}_Avail"] = 1 if raw_avail else 0
            row[f"{sid}_Pref"]  = int(input[f"t_pref_{i}_{sid}"]())
        rows.append(row)
    return pd.DataFrame(rows) if rows else None


def build_shift_df_from_inputs(input, open_days):
    rows = []
    for day in open_days:
        for st in SHIFT_TYPES:
            sid = f"{day}_{st}"
            mgr  = int(input[f"s_mgr_{sid}"]())
            lead = int(input[f"s_lead_{sid}"]())
            srv  = int(input[f"s_srv_{sid}"]())
            host = int(input[f"s_host_{sid}"]())
            rows.append({
                "Shift_ID":    sid,
                "Day":         day,
                "Shift_Type":  st,
                "Total_Staff": mgr + lead + srv + host,
                "Manager":     mgr,
                "Lead_Server": lead,
                "Server":      srv,
                "Host":        host,
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  METRICS
# ─────────────────────────────────────────────

def compute_metrics(sched_df, summ_df, shift_df):
    if sched_df.empty or summ_df.empty:
        return {}
    hours          = summ_df["Hours"].tolist()
    total_slots    = shift_df["Total_Staff"].sum() if "Total_Staff" in shift_df.columns else 0
    pref_score     = summ_df["Pref_Score"].sum()     if "Pref_Score"     in summ_df.columns else None
    max_pref_score = summ_df["Max_Pref_Score"].sum() if "Max_Pref_Score" in summ_df.columns else None
    pref_pct = round(100 * pref_score / max_pref_score, 1) if (pref_score and max_pref_score) else None
    avg_h = round(sum(hours) / len(hours), 1) if hours else 0
    std_h = round(pd.Series(hours).std(), 1)  if len(hours) > 1 else 0
    return {
        "Shifts Scheduled":         len(shift_df),
        "Total Staff Slots":        int(total_slots),
        "Avg Hours / Employee":     avg_h,
        "Max Hours (any emp)":      max(hours) if hours else 0,
        "Min Hours (any emp)":      min(hours) if hours else 0,
        "Hours Std Dev (Fairness)": std_h,
        "Pref Satisfaction":        f"{pref_pct}%" if pref_pct is not None else "N/A",
        "Staffing Coverage":        "100%",
    }


# ─────────────────────────────────────────────
#  CALLOUT RESOLVER
# ─────────────────────────────────────────────

def _parse_worker_names(raw):
    names = set()
    if not raw or isinstance(raw, float):
        return names
    for w in str(raw).split(","):
        w = w.strip()
        if "(" in w:
            w = w[:w.rfind("(")].strip()
        if w:
            names.add(w)
    return names


def _rebuild_summary(emp_df, shift_df, sched_df, roles, shift_type):
    employees = emp_df["Name"].tolist()
    shift_ids = shift_df["Shift_ID"].tolist()
    amap = {}
    for _, row in sched_df.iterrows():
        amap[row["Shift"]] = _parse_worker_names(row.get("Workers", ""))
    summ_rows = []
    for e in employees:
        assigned = [s for s in shift_ids if e in amap.get(s, set())]
        n        = len(assigned)
        hours    = n * HOURS_PER_SHIFT
        pref_score = 0
        for s in assigned:
            pc = f"{s}_Pref"
            if pc in emp_df.columns:
                pref_score += int(pd.to_numeric(
                    emp_df.loc[emp_df["Name"] == e, pc], errors="coerce").fillna(1).values[0])
        all_prefs = []
        for s in shift_ids:
            pc = f"{s}_Pref"
            if pc in emp_df.columns:
                all_prefs.append(int(pd.to_numeric(
                    emp_df.loc[emp_df["Name"] == e, pc], errors="coerce").fillna(1).values[0]))
        max_pref = sum(sorted(all_prefs, reverse=True)[:n]) if n else 0
        summ_rows.append({
            "Name":            e,
            "Role":            roles.get(e, ""),
            "Shifts":          n,
            "Hours":           hours,
            "AM Shifts":       sum(1 for s in assigned if shift_type.get(s) == "AM"),
            "PM Shifts":       sum(1 for s in assigned if shift_type.get(s) == "PM"),
            "Pref_Score":      pref_score,
            "Max_Pref_Score":  max_pref,
            "Pref %":          f"{round(100*pref_score/max_pref)}%" if max_pref else "N/A",
            "Assigned Shifts": ", ".join(assigned),
        })
    return pd.DataFrame(summ_rows)


def resolve_callout(emp_df, shift_df, current_sched_df, absent_emp,
                    affected_shift_id, constraints, blocked_pairs=None):
    if blocked_pairs is None:
        blocked_pairs = set()

    emp_df   = clean_emp(emp_df)
    shift_df = clean_shift(shift_df)

    if "Name" not in emp_df.columns:
        return pd.DataFrame(), pd.DataFrame(), ["Employee file missing 'Name' column."]

    employees  = emp_df["Name"].tolist()
    shift_ids  = shift_df["Shift_ID"].tolist()
    roles      = dict(zip(employees, emp_df["Role"]))
    max_hours  = dict(zip(employees, emp_df["Max_Hours"]))
    shift_type = dict(zip(shift_df["Shift_ID"], shift_df["Shift_Type"]))
    shift_day  = dict(zip(shift_df["Shift_ID"], shift_df["Day"]))
    role_map   = {"Manager":"Manager","Lead_Server":"Lead Server","Server":"Server","Host":"Host"}

    existing = {}
    for _, row in current_sched_df.iterrows():
        existing[row["Shift"]] = _parse_worker_names(row.get("Workers", ""))

    remaining    = existing.get(affected_shift_id, set()) - {absent_emp}
    shift_row_df = shift_df[shift_df["Shift_ID"] == affected_shift_id]
    if shift_row_df.empty:
        return pd.DataFrame(), pd.DataFrame(), [f"Shift '{affected_shift_id}' not found."]
    shift_row    = shift_row_df.iloc[0]
    total_needed = int(shift_row["Total_Staff"])

    if total_needed - len(remaining) <= 0:
        updated = current_sched_df.copy()
        updated.loc[updated["Shift"] == affected_shift_id, "Workers"] = \
            ", ".join(f"{e} ({roles.get(e,'')})" for e in sorted(remaining))
        updated.loc[updated["Shift"] == affected_shift_id, "# Staff"] = len(remaining)
        return updated, _rebuild_summary(emp_df, shift_df, updated, roles, shift_type), []

    hours_elsewhere = {e: 0 for e in employees}
    for sid, wset in existing.items():
        if sid != affected_shift_id:
            for e in wset:
                if e in hours_elsewhere:
                    hours_elsewhere[e] += HOURS_PER_SHIFT

    affected_day  = shift_day.get(affected_shift_id, "")
    affected_type = shift_type.get(affected_shift_id, "")

    busy_same_day = set()
    for sid, wset in existing.items():
        if sid != affected_shift_id and shift_day.get(sid) == affected_day:
            busy_same_day.update(wset)

    clopening_blocked = set()
    if constraints.get("no_clopening", True):
        day_index = {d: i for i, d in enumerate(DAY_ORDER)}
        aff_idx   = day_index.get(affected_day, -1)
        for e in employees:
            if affected_type == "PM" and 0 <= aff_idx < len(DAY_ORDER) - 1:
                next_am = [s for s in shift_ids
                           if shift_day.get(s) == DAY_ORDER[aff_idx+1] and shift_type.get(s) == "AM"]
                if any(e in existing.get(s, set()) for s in next_am):
                    clopening_blocked.add(e)
            if affected_type == "AM" and aff_idx > 0:
                prev_pm = [s for s in shift_ids
                           if shift_day.get(s) == DAY_ORDER[aff_idx-1] and shift_type.get(s) == "PM"]
                if any(e in existing.get(s, set()) for s in prev_pm):
                    clopening_blocked.add(e)

    avail_col  = f"{affected_shift_id}_Avail"
    candidates = []
    for e in employees:
        if e == absent_emp or e in remaining:
            continue
        if (e, affected_shift_id) in blocked_pairs:
            continue
        if e in busy_same_day:
            continue
        if constraints.get("no_clopening", True) and e in clopening_blocked:
            continue
        if constraints.get("availability", True) and avail_col in emp_df.columns:
            val = pd.to_numeric(emp_df.loc[emp_df["Name"]==e, avail_col],
                                errors="coerce").fillna(0).values[0]
            if int(val) == 0:
                continue
        if constraints.get("max_hours", True):
            if hours_elsewhere[e] + HOURS_PER_SHIFT > max_hours[e]:
                continue
        candidates.append(e)

    if not candidates:
        return pd.DataFrame(), pd.DataFrame(), [
            f"❌ No replacement found for {absent_emp} on {affected_shift_id} — "
            f"all remaining employees violate at least one active constraint "
            f"(availability, max hours, same-day shift, or no-clopening rule). "
            f"Try unchecking a constraint and re-optimizing."
        ]

    role_needs = {}
    for col, emp_role in role_map.items():
        if col in shift_row and int(shift_row[col]) > 0:
            needed  = int(shift_row[col])
            already = sum(1 for e in remaining
                          if emp_role in ROLE_HIERARCHY.get(roles.get(e,""), []))
            if needed - already > 0:
                role_needs[emp_role] = needed - already

    selected = list(remaining)

    def priority(e):
        pc = f"{affected_shift_id}_Pref"
        pref = 1
        if pc in emp_df.columns:
            pref = int(pd.to_numeric(emp_df.loc[emp_df["Name"]==e, pc],
                                     errors="coerce").fillna(1).values[0])
        fills = any(r in ROLE_HIERARCHY.get(roles.get(e,""), []) for r in role_needs)
        return (0 if fills else 1, -pref)

    candidates.sort(key=priority)
    for e in candidates:
        if len(selected) >= total_needed:
            break
        selected.append(e)

    unmet = []
    for col, emp_role in role_map.items():
        if col in shift_row and int(shift_row[col]) > 0:
            needed  = int(shift_row[col])
            covered = sum(1 for e in selected
                          if emp_role in ROLE_HIERARCHY.get(roles.get(e,""), []))
            if covered < needed:
                unmet.append(f"{needed}×{emp_role} (only {covered})")
    if unmet:
        return pd.DataFrame(), pd.DataFrame(), [
            f"❌ Cannot fill role requirements for {affected_shift_id}: " + ", ".join(unmet)
        ]

    updated = current_sched_df.copy()
    updated.loc[updated["Shift"] == affected_shift_id, "Workers"] = \
        ", ".join(f"{e} ({roles.get(e,'')})" for e in selected)
    updated.loc[updated["Shift"] == affected_shift_id, "# Staff"] = len(selected)
    return updated, _rebuild_summary(emp_df, shift_df, updated, roles, shift_type), []


# ─────────────────────────────────────────────
#  OPTIMIZATION ENGINE
# ─────────────────────────────────────────────

def build_schedule(emp_df, shift_df, constraints):
    emp_df   = clean_emp(emp_df)
    shift_df = clean_shift(shift_df)

    if "Name" not in emp_df.columns:
        return pd.DataFrame(), pd.DataFrame(), ["Employee file missing 'Name' column."]
    for col in ["Shift_ID", "Day", "Shift_Type", "Total_Staff"]:
        if col not in shift_df.columns:
            return pd.DataFrame(), pd.DataFrame(), [f"Shift file missing '{col}' column."]

    employees  = emp_df["Name"].tolist()
    shift_ids  = shift_df["Shift_ID"].tolist()
    roles      = dict(zip(employees, emp_df["Role"]))
    max_hours  = dict(zip(employees, emp_df["Max_Hours"]))
    shift_type = dict(zip(shift_df["Shift_ID"], shift_df["Shift_Type"]))
    role_map   = {"Manager":"Manager","Lead_Server":"Lead Server","Server":"Server","Host":"Host"}
    all_roles  = list(role_map.values())

    if len(employees) == 0:
        return pd.DataFrame(), pd.DataFrame(), [
            "No employees provided. Add at least one employee before generating a schedule."
        ]

    # Pre-audit
    audit_errors = []
    for _, row in shift_df.iterrows():
        sid = row["Shift_ID"]
        for col, emp_role in role_map.items():
            if col not in row or int(row[col]) == 0:
                continue
            needed    = int(row[col])
            avail_col = f"{sid}_Avail"
            if avail_col in emp_df.columns:
                q = [e for e in employees
                     if emp_role in ROLE_HIERARCHY.get(roles[e], [])
                     and int(pd.to_numeric(emp_df.loc[emp_df["Name"]==e, avail_col],
                                           errors="coerce").fillna(0).values[0]) == 1]
            else:
                q = [e for e in employees if emp_role in ROLE_HIERARCHY.get(roles[e], [])]
            if len(q) < needed:
                audit_errors.append(
                    f"INFEASIBLE: {sid} needs {needed}×{emp_role} "
                    f"but only {len(q)} qualified & available.")
    if audit_errors:
        return pd.DataFrame(), pd.DataFrame(), audit_errors

    model = cp_model.CpModel()

    required_roles = {}
    for _, row in shift_df.iterrows():
        sid = row["Shift_ID"]
        required_roles[sid] = [
            (emp_role, int(row[col]))
            for col, emp_role in role_map.items()
            if col in row and int(row[col]) > 0
        ]

    y = {(e, s, r): model.NewBoolVar(f"y_{e}_{s}_{r}")
         for e in employees for s in shift_ids for r in all_roles}
    x = {(e, s): model.NewBoolVar(f"x_{e}_{s}")
         for e in employees for s in shift_ids}

    for e in employees:
        for s in shift_ids:
            model.Add(sum(y[e, s, r] for r in all_roles) == x[e, s])
            for r in all_roles:
                if r not in ROLE_HIERARCHY.get(roles[e], []):
                    model.Add(y[e, s, r] == 0)

    for _, row in shift_df.iterrows():
        sid = row["Shift_ID"]
        for emp_role, needed in required_roles[sid]:
            model.Add(sum(y[e, sid, emp_role] for e in employees) == needed)
        model.Add(sum(x[e, sid] for e in employees) == int(row["Total_Staff"]))

    if constraints.get("availability", True):
        for e in employees:
            for s in shift_ids:
                ac = f"{s}_Avail"
                if ac in emp_df.columns:
                    val = pd.to_numeric(emp_df.loc[emp_df["Name"]==e, ac],
                                        errors="coerce").fillna(0).values[0]
                    if int(val) == 0:
                        model.Add(x[e, s] == 0)

    if constraints.get("max_hours", True):
        for e in employees:
            model.Add(sum(x[e, s] for s in shift_ids) * HOURS_PER_SHIFT <= max_hours[e])

    for d in shift_df["Day"].unique():
        day_shifts = shift_df[shift_df["Day"] == d]["Shift_ID"].tolist()
        for e in employees:
            model.Add(sum(x[e, s] for s in day_shifts) <= 1)

    if constraints.get("no_clopening", True):
        day_index = {d: i for i, d in enumerate(DAY_ORDER)}
        for _, row_pm in shift_df[shift_df["Shift_Type"] == "PM"].iterrows():
            pm_sid = row_pm["Shift_ID"]
            pm_idx = day_index.get(row_pm["Day"], -1)
            if pm_idx < 0 or pm_idx + 1 >= len(DAY_ORDER):
                continue
            next_am = shift_df[
                (shift_df["Day"] == DAY_ORDER[pm_idx+1]) &
                (shift_df["Shift_Type"] == "AM")
            ]["Shift_ID"].tolist()
            for am_sid in next_am:
                for e in employees:
                    model.Add(x[e, pm_sid] + x[e, am_sid] <= 1)

    role_to_col = {v: k for k, v in role_map.items()}
    if constraints.get("fairness", True):
        for e in employees:
            emp_role    = roles[e]
            primary_col = role_to_col.get(emp_role)
            primary_slots = (int(shift_df[primary_col].sum())
                             if primary_col and primary_col in shift_df.columns
                             else len(shift_ids))
            peers   = [p for p in employees if roles[p] == emp_role]
            avg     = primary_slots // len(peers)
            fair_min = max(0, avg - 1)
            fair_max = min(len(shift_ids), avg + 1)
            sw = sum(x[e, s] for s in shift_ids)
            model.Add(sw >= fair_min)
            model.Add(sw <= fair_max)

    avg_int = int(shift_df["Total_Staff"].sum()) // len(employees)
    pref_terms     = []
    fairness_terms = []
    for e in employees:
        sw_var = model.NewIntVar(0, len(shift_ids), f"sw_{e}")
        model.Add(sw_var == sum(x[e, s] for s in shift_ids))
        diff_var = model.NewIntVar(-len(shift_ids), len(shift_ids), f"diff_{e}")
        model.Add(diff_var == sw_var - avg_int)
        over_var = model.NewIntVar(0, len(shift_ids), f"over_{e}")
        model.AddMaxEquality(over_var, [diff_var, model.NewConstant(0)])
        fairness_terms.append(over_var)
        for s in shift_ids:
            pc = f"{s}_Pref"
            if pc in emp_df.columns:
                w = int(pd.to_numeric(emp_df.loc[emp_df["Name"]==e, pc],
                                      errors="coerce").fillna(1).values[0])
                pref_terms.append(x[e, s] * w * 10)

    objective = pref_terms + [-5 * t for t in fairness_terms]
    if objective:
        model.Maximize(sum(objective))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # Run targeted diagnosis to give the manager a specific, actionable error
        diag = []

        # Check 1: role supply per shift after all constraints applied
        for _, row in shift_df.iterrows():
            sid = row["Shift_ID"]
            for col, emp_role in role_map.items():
                if col not in row or int(row[col]) == 0:
                    continue
                needed    = int(row[col])
                avail_col = f"{sid}_Avail"
                # Qualified + available
                if constraints.get("availability", True) and avail_col in emp_df.columns:
                    pool = [e for e in employees
                            if emp_role in ROLE_HIERARCHY.get(roles[e], [])
                            and int(pd.to_numeric(emp_df.loc[emp_df["Name"]==e, avail_col],
                                                  errors="coerce").fillna(0).values[0]) == 1]
                else:
                    pool = [e for e in employees if emp_role in ROLE_HIERARCHY.get(roles[e], [])]
                if len(pool) < needed:
                    diag.append(
                        f"❌ {sid} needs {needed} × {emp_role} "
                        f"but only {len(pool)} qualified employee(s) are available for that shift."
                    )

        # Check 2: fairness floor impossible for a role group
        if constraints.get("fairness", True):
            role_to_col = {v: k for k, v in role_map.items()}
            for base_role in set(roles.values()):
                peers       = [e for e in employees if roles[e] == base_role]
                primary_col = role_to_col.get(base_role)
                if primary_col and primary_col in shift_df.columns:
                    slots = int(shift_df[primary_col].sum())
                    avg   = slots // len(peers)
                    fair_min = max(0, avg - 1)
                    if fair_min * len(peers) > slots:
                        diag.append(
                            f"❌ Fairness floor infeasible for {base_role}: "
                            f"{len(peers)} employees × {fair_min} min shifts = "
                            f"{fair_min*len(peers)} required, but only {slots} {base_role} slots exist. "
                            f"Try unchecking Fairness."
                        )

        # Check 3: max hours too low for the schedule
        if constraints.get("max_hours", True):
            total_slots = int(shift_df["Total_Staff"].sum())
            total_capacity = sum(max_hours[e] // HOURS_PER_SHIFT for e in employees)
            if total_capacity < total_slots:
                diag.append(
                    f"❌ Max-hours constraint too tight: employees can cover at most "
                    f"{total_capacity} shifts total, but {total_slots} shifts need filling. "
                    f"Increase max hours or add more employees."
                )

        if not diag:
            diag = [
                "❌ Solver could not find a valid schedule. "
                "Try unchecking one or more constraints (Availability, Fairness, or No Clopening) "
                "to identify which constraint is causing the conflict."
            ]
        return pd.DataFrame(), pd.DataFrame(), diag

    sched_rows = []
    for _, row in shift_df.iterrows():
        s = row["Shift_ID"]
        workers = []
        for e in employees:
            if solver.Value(x[e, s]) == 1:
                filled = roles[e]
                for r in all_roles:
                    if solver.Value(y[e, s, r]) == 1:
                        filled = r
                        break
                workers.append(f"{e} ({filled})")
        sched_rows.append({
            "Shift":      s,
            "Day":        row["Day"],
            "Shift Type": row["Shift_Type"],
            "# Staff":    len(workers),
            "Workers":    ", ".join(workers),
        })

    summ_rows = []
    for e in employees:
        assigned = [s for s in shift_ids if solver.Value(x[e, s]) == 1]
        n        = len(assigned)
        hours    = n * HOURS_PER_SHIFT
        pref_score = 0
        for s in assigned:
            pc = f"{s}_Pref"
            if pc in emp_df.columns:
                pref_score += int(pd.to_numeric(
                    emp_df.loc[emp_df["Name"]==e, pc], errors="coerce").fillna(1).values[0])
        all_prefs = []
        for s in shift_ids:
            pc = f"{s}_Pref"
            if pc in emp_df.columns:
                all_prefs.append(int(pd.to_numeric(
                    emp_df.loc[emp_df["Name"]==e, pc], errors="coerce").fillna(1).values[0]))
        max_pref = sum(sorted(all_prefs, reverse=True)[:n]) if n else 0
        summ_rows.append({
            "Name":            e,
            "Role":            roles[e],
            "Shifts":          n,
            "Hours":           hours,
            "AM Shifts":       sum(1 for s in assigned if shift_type.get(s) == "AM"),
            "PM Shifts":       sum(1 for s in assigned if shift_type.get(s) == "PM"),
            "Pref_Score":      pref_score,
            "Max_Pref_Score":  max_pref,
            "Pref %":          f"{round(100*pref_score/max_pref)}%" if max_pref else "N/A",
            "Assigned Shifts": ", ".join(assigned),
        })

    return pd.DataFrame(sched_rows), pd.DataFrame(summ_rows), []


# ─────────────────────────────────────────────
#  AI CHAT HELPER
# ─────────────────────────────────────────────

# ── Paste your Anthropic API key here ──────────────────────────────────────
ANTHROPIC_API_KEY = "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"
# ───────────────────────────────────────────────────────────────────────────

def ask_schedule_ai(user_question, sched_df, summ_df, metrics, change_log_text):
    """Call Claude to answer manager questions about the current schedule."""
    if _anthropic_mod is None:
        return "❌ The anthropic package is not installed. Run: pip install anthropic"
    if not ANTHROPIC_API_KEY or ANTHROPIC_API_KEY == "your-api-key-here":
        return "❌ Add your Anthropic API key to the ANTHROPIC_API_KEY variable at the top of the file."
    if sched_df is None or sched_df.empty:
        return "❌ No schedule generated yet. Please generate a schedule first."

    metrics_text = "\n".join(f"  {k}: {v}" for k, v in metrics.items()) if metrics else "None"

    prompt = f"""You are a helpful restaurant scheduling assistant for managers.

Answer ONLY using the schedule data provided below.
Do not invent employees, shifts, or hours not shown in the data.
If the answer is not in the data, say so clearly.
Be concise and manager-friendly.

WEEKLY SCHEDULE:
{sched_df.to_string(index=False)}

EMPLOYEE SUMMARY:
{summ_df.to_string(index=False) if summ_df is not None and not summ_df.empty else "Not available"}

SCHEDULE METRICS:
{metrics_text}

LATEST CALL-OUT CHANGE:
{change_log_text if change_log_text else "None"}

MANAGER QUESTION:
{user_question}"""

    try:
        client = _anthropic_mod.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.content[0].text
    except Exception as e:
        return f"❌ API error: {e}"


DEFAULT_EMP_DATA = [
    {"name": "Maggie", "role": "Manager", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "4", "avail_Mon_PM": 1, "pref_Mon_PM": "2", "avail_Tue_AM": 1, "pref_Tue_AM": "5", "avail_Tue_PM": 1, "pref_Tue_PM": "4", "avail_Wed_AM": 1, "pref_Wed_AM": "1", "avail_Wed_PM": 1, "pref_Wed_PM": "4", "avail_Thu_AM": 1, "pref_Thu_AM": "5", "avail_Thu_PM": 1, "pref_Thu_PM": "1", "avail_Fri_AM": 1, "pref_Fri_AM": "5", "avail_Fri_PM": 1, "pref_Fri_PM": "1", "avail_Sat_AM": 1, "pref_Sat_AM": "2", "avail_Sat_PM": 1, "pref_Sat_PM": "4"},
    {"name": "Morgan", "role": "Manager", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "1", "avail_Mon_PM": 1, "pref_Mon_PM": "4", "avail_Tue_AM": 1, "pref_Tue_AM": "3", "avail_Tue_PM": 1, "pref_Tue_PM": "3", "avail_Wed_AM": 1, "pref_Wed_AM": "1", "avail_Wed_PM": 1, "pref_Wed_PM": "2", "avail_Thu_AM": 1, "pref_Thu_AM": "4", "avail_Thu_PM": 1, "pref_Thu_PM": "3", "avail_Fri_AM": 1, "pref_Fri_AM": "2", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "1", "avail_Sat_PM": 1, "pref_Sat_PM": "1"},
    {"name": "Sandra", "role": "Manager", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "1", "avail_Mon_PM": 1, "pref_Mon_PM": "2", "avail_Tue_AM": 1, "pref_Tue_AM": "4", "avail_Tue_PM": 1, "pref_Tue_PM": "2", "avail_Wed_AM": 1, "pref_Wed_AM": "3", "avail_Wed_PM": 1, "pref_Wed_PM": "1", "avail_Thu_AM": 1, "pref_Thu_AM": "3", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "2", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "4", "avail_Sat_PM": 1, "pref_Sat_PM": "5"},
    {"name": "Steve", "role": "Lead Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "3", "avail_Mon_PM": 1, "pref_Mon_PM": "2", "avail_Tue_AM": 1, "pref_Tue_AM": "3", "avail_Tue_PM": 1, "pref_Tue_PM": "4", "avail_Wed_AM": 1, "pref_Wed_AM": "2", "avail_Wed_PM": 1, "pref_Wed_PM": "4", "avail_Thu_AM": 1, "pref_Thu_AM": "1", "avail_Thu_PM": 1, "pref_Thu_PM": "1", "avail_Fri_AM": 1, "pref_Fri_AM": "1", "avail_Fri_PM": 1, "pref_Fri_PM": "1", "avail_Sat_AM": 1, "pref_Sat_AM": "3", "avail_Sat_PM": 1, "pref_Sat_PM": "5"},
    {"name": "Sarah", "role": "Lead Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "3", "avail_Mon_PM": 1, "pref_Mon_PM": "4", "avail_Tue_AM": 1, "pref_Tue_AM": "3", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "5", "avail_Wed_PM": 1, "pref_Wed_PM": "4", "avail_Thu_AM": 1, "pref_Thu_AM": "1", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "4", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "4", "avail_Sat_PM": 1, "pref_Sat_PM": "1"},
    {"name": "Brian", "role": "Lead Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "2", "avail_Mon_PM": 1, "pref_Mon_PM": "4", "avail_Tue_AM": 1, "pref_Tue_AM": "1", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "4", "avail_Wed_PM": 1, "pref_Wed_PM": "1", "avail_Thu_AM": 1, "pref_Thu_AM": "4", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "2", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "4", "avail_Sat_PM": 1, "pref_Sat_PM": "4"},
    {"name": "Brenda", "role": "Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "4", "avail_Mon_PM": 1, "pref_Mon_PM": "1", "avail_Tue_AM": 1, "pref_Tue_AM": "3", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "2", "avail_Wed_PM": 1, "pref_Wed_PM": "2", "avail_Thu_AM": 1, "pref_Thu_AM": "3", "avail_Thu_PM": 1, "pref_Thu_PM": "3", "avail_Fri_AM": 1, "pref_Fri_AM": "3", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "2", "avail_Sat_PM": 1, "pref_Sat_PM": "2"},
    {"name": "Billy", "role": "Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "4", "avail_Mon_PM": 1, "pref_Mon_PM": "5", "avail_Tue_AM": 1, "pref_Tue_AM": "5", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "4", "avail_Wed_PM": 1, "pref_Wed_PM": "1", "avail_Thu_AM": 1, "pref_Thu_AM": "3", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "4", "avail_Fri_PM": 1, "pref_Fri_PM": "3", "avail_Sat_AM": 1, "pref_Sat_AM": "3", "avail_Sat_PM": 1, "pref_Sat_PM": "5"},
    {"name": "Bo", "role": "Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "3", "avail_Mon_PM": 1, "pref_Mon_PM": "5", "avail_Tue_AM": 1, "pref_Tue_AM": "3", "avail_Tue_PM": 1, "pref_Tue_PM": "3", "avail_Wed_AM": 1, "pref_Wed_AM": "3", "avail_Wed_PM": 1, "pref_Wed_PM": "4", "avail_Thu_AM": 1, "pref_Thu_AM": "5", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "5", "avail_Fri_PM": 1, "pref_Fri_PM": "4", "avail_Sat_AM": 1, "pref_Sat_AM": "4", "avail_Sat_PM": 1, "pref_Sat_PM": "1"},
    {"name": "Bailey", "role": "Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "4", "avail_Mon_PM": 1, "pref_Mon_PM": "2", "avail_Tue_AM": 1, "pref_Tue_AM": "1", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "3", "avail_Wed_PM": 1, "pref_Wed_PM": "5", "avail_Thu_AM": 1, "pref_Thu_AM": "3", "avail_Thu_PM": 1, "pref_Thu_PM": "3", "avail_Fri_AM": 1, "pref_Fri_AM": "1", "avail_Fri_PM": 1, "pref_Fri_PM": "2", "avail_Sat_AM": 1, "pref_Sat_AM": "1", "avail_Sat_PM": 1, "pref_Sat_PM": "3"},
    {"name": "Hector", "role": "Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "4", "avail_Mon_PM": 1, "pref_Mon_PM": "5", "avail_Tue_AM": 1, "pref_Tue_AM": "5", "avail_Tue_PM": 1, "pref_Tue_PM": "4", "avail_Wed_AM": 1, "pref_Wed_AM": "1", "avail_Wed_PM": 1, "pref_Wed_PM": "5", "avail_Thu_AM": 1, "pref_Thu_AM": "3", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "5", "avail_Fri_PM": 1, "pref_Fri_PM": "3", "avail_Sat_AM": 1, "pref_Sat_AM": "1", "avail_Sat_PM": 1, "pref_Sat_PM": "1"},
    {"name": "Holly", "role": "Server", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "1", "avail_Mon_PM": 1, "pref_Mon_PM": "2", "avail_Tue_AM": 1, "pref_Tue_AM": "2", "avail_Tue_PM": 1, "pref_Tue_PM": "5", "avail_Wed_AM": 1, "pref_Wed_AM": "5", "avail_Wed_PM": 1, "pref_Wed_PM": "3", "avail_Thu_AM": 1, "pref_Thu_AM": "3", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "4", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "5", "avail_Sat_PM": 1, "pref_Sat_PM": "3"},
    {"name": "John", "role": "Host", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "3", "avail_Mon_PM": 1, "pref_Mon_PM": "1", "avail_Tue_AM": 1, "pref_Tue_AM": "3", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "4", "avail_Wed_PM": 1, "pref_Wed_PM": "1", "avail_Thu_AM": 1, "pref_Thu_AM": "2", "avail_Thu_PM": 1, "pref_Thu_PM": "2", "avail_Fri_AM": 1, "pref_Fri_AM": "4", "avail_Fri_PM": 1, "pref_Fri_PM": "1", "avail_Sat_AM": 1, "pref_Sat_AM": "3", "avail_Sat_PM": 1, "pref_Sat_PM": "4"},
    {"name": "Jane", "role": "Host", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "5", "avail_Mon_PM": 1, "pref_Mon_PM": "4", "avail_Tue_AM": 1, "pref_Tue_AM": "1", "avail_Tue_PM": 1, "pref_Tue_PM": "3", "avail_Wed_AM": 1, "pref_Wed_AM": "2", "avail_Wed_PM": 1, "pref_Wed_PM": "1", "avail_Thu_AM": 1, "pref_Thu_AM": "5", "avail_Thu_PM": 1, "pref_Thu_PM": "1", "avail_Fri_AM": 1, "pref_Fri_AM": "4", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "3", "avail_Sat_PM": 1, "pref_Sat_PM": "2"},
    {"name": "Kevin", "role": "Host", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "3", "avail_Mon_PM": 1, "pref_Mon_PM": "4", "avail_Tue_AM": 1, "pref_Tue_AM": "2", "avail_Tue_PM": 1, "pref_Tue_PM": "3", "avail_Wed_AM": 1, "pref_Wed_AM": "3", "avail_Wed_PM": 1, "pref_Wed_PM": "3", "avail_Thu_AM": 1, "pref_Thu_AM": "1", "avail_Thu_PM": 1, "pref_Thu_PM": "1", "avail_Fri_AM": 1, "pref_Fri_AM": "5", "avail_Fri_PM": 1, "pref_Fri_PM": "4", "avail_Sat_AM": 1, "pref_Sat_AM": "5", "avail_Sat_PM": 1, "pref_Sat_PM": "1"},
    {"name": "Kim", "role": "Host", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "5", "avail_Mon_PM": 1, "pref_Mon_PM": "4", "avail_Tue_AM": 1, "pref_Tue_AM": "2", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "1", "avail_Wed_PM": 1, "pref_Wed_PM": "3", "avail_Thu_AM": 1, "pref_Thu_AM": "4", "avail_Thu_PM": 1, "pref_Thu_PM": "1", "avail_Fri_AM": 1, "pref_Fri_AM": "4", "avail_Fri_PM": 1, "pref_Fri_PM": "5", "avail_Sat_AM": 1, "pref_Sat_AM": "4", "avail_Sat_PM": 1, "pref_Sat_PM": "5"},
    {"name": "Leo", "role": "Host", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "1", "avail_Mon_PM": 1, "pref_Mon_PM": "5", "avail_Tue_AM": 1, "pref_Tue_AM": "4", "avail_Tue_PM": 1, "pref_Tue_PM": "5", "avail_Wed_AM": 1, "pref_Wed_AM": "1", "avail_Wed_PM": 1, "pref_Wed_PM": "3", "avail_Thu_AM": 1, "pref_Thu_AM": "1", "avail_Thu_PM": 1, "pref_Thu_PM": "3", "avail_Fri_AM": 1, "pref_Fri_AM": "5", "avail_Fri_PM": 1, "pref_Fri_PM": "1", "avail_Sat_AM": 1, "pref_Sat_AM": "3", "avail_Sat_PM": 1, "pref_Sat_PM": "4"},
    {"name": "Lily", "role": "Host", "max_hours": 40, "avail_Mon_AM": 1, "pref_Mon_AM": "2", "avail_Mon_PM": 1, "pref_Mon_PM": "1", "avail_Tue_AM": 1, "pref_Tue_AM": "5", "avail_Tue_PM": 1, "pref_Tue_PM": "1", "avail_Wed_AM": 1, "pref_Wed_AM": "4", "avail_Wed_PM": 1, "pref_Wed_PM": "4", "avail_Thu_AM": 1, "pref_Thu_AM": "5", "avail_Thu_PM": 1, "pref_Thu_PM": "5", "avail_Fri_AM": 1, "pref_Fri_AM": "4", "avail_Fri_PM": 1, "pref_Fri_PM": "4", "avail_Sat_AM": 1, "pref_Sat_AM": "1", "avail_Sat_PM": 1, "pref_Sat_PM": "1"},
]

# ─────────────────────────────────────────────
#  TEMPLATE FORM HELPERS
# ─────────────────────────────────────────────

def make_shift_form():
    """
    Shift Requirements page. Each day has an Open toggle.
    When open, AM and PM staffing inputs appear. When closed, they collapse.
    The open_{day} checkbox IDs are read by get_open_days() in the server.
    """
    col_hdr = "flex:0 0 55px; font-weight:600; font-size:12px; color:#495057; margin-left:6px;"
    col_hdr_c = col_hdr + " text-align:center;"

    table_header = ui.div(
        ui.div("",       style="flex:0 0 24px;"),                    # open checkbox
        ui.div("Shift",  style="flex:0 0 80px; font-weight:600; font-size:12px; color:#495057; margin-left:6px;"),
        ui.div("Total",  style=col_hdr_c),
        ui.div("Mgr",    style=col_hdr),
        ui.div("Lead",   style=col_hdr),
        ui.div("Server", style=col_hdr),
        ui.div("Host",   style=col_hdr),
        style="display:flex; align-items:center; padding:4px 0; "
              "border-bottom:2px solid #dee2e6; margin-bottom:4px; color:#6c757d;"
    )

    day_blocks = [table_header]

    for day in ALL_DAYS:
        is_default_open = (day != "Sun")

        # Day header row with open/closed toggle
        day_header = ui.div(
            ui.input_checkbox(f"open_{day}", "", value=is_default_open),
            ui.div(
                ui.tags.b(day),
                ui.span(" — Open" if is_default_open else " — Closed",
                        id=f"open_label_{day}",
                        style="font-size:12px; color:#6c757d; margin-left:4px;"),
                style="margin-left:6px; font-size:14px;"
            ),
            style="display:flex; align-items:center; padding:6px 0 4px; "
                  "border-top:2px solid #e9ecef; margin-top:4px;"
        )

        # AM and PM shift input rows for this day
        shift_rows = []
        for st in SHIFT_TYPES:
            sid = f"{day}_{st}"
            shift_rows.append(ui.div(
                ui.div("",    style="flex:0 0 24px;"),               # spacer
                ui.div(f"  {st}", style="flex:0 0 80px; font-size:13px; font-weight:500; "
                                        "color:#495057; margin-left:6px;"),
                ui.div(ui.output_ui(f"total_{sid}"),
                       style="flex:0 0 55px; margin-left:6px; text-align:center;"),
                ui.div(ui.input_numeric(f"s_mgr_{sid}",  None, value=1, min=0, max=20, width="48px"),
                       style="flex:0 0 55px; margin-left:6px;"),
                ui.div(ui.input_numeric(f"s_lead_{sid}", None, value=1, min=0, max=20, width="48px"),
                       style="flex:0 0 55px; margin-left:6px;"),
                ui.div(ui.input_numeric(f"s_srv_{sid}",  None, value=3, min=0, max=20, width="48px"),
                       style="flex:0 0 55px; margin-left:6px;"),
                ui.div(ui.input_numeric(f"s_host_{sid}", None, value=1, min=0, max=20, width="48px"),
                       style="flex:0 0 55px; margin-left:6px;"),
                style="display:flex; align-items:center; padding:3px 0; "
                      "border-bottom:1px solid #f4f4f4;"
            ))

        # Wrap shift rows in a collapsible container
        shift_container = ui.div(
            *shift_rows,
            id=f"shift_rows_{day}",
            style="" if is_default_open else "display:none;"
        )

        day_blocks.append(ui.div(day_header, shift_container))

    return ui.div(*day_blocks)


def make_employee_table(n, open_days=None, defaults=None):
    if open_days is None:
        open_days = ALL_DAYS
    if defaults is None:
        defaults = []
    """
    Build the ENTIRE employee template table — header + all rows — as one
    static Shiny UI tree. This avoids injecting <tr> into a <table> via
    output_ui, which breaks because Shiny wraps dynamic output in a <div>.
    """
    # 3-row header:
    #   Row 1: Name | Role | Hrs | Mon (colspan=4) | Tue (colspan=4) | ...
    #   Row 2: (empty x3) | AM (colspan=2) | PM (colspan=2) | AM ...
    #   Row 3: (empty x3) | ✔ | ★ | ✔ | ★ | ...
    th_base = ("padding:3px 5px; font-size:11px; font-weight:600; white-space:nowrap; "
               "border-bottom:2px solid #dee2e6; background:#fff;")
    th_day  = ("padding:3px 4px; font-size:11px; font-weight:600; text-align:center; "
               "border-bottom:1px solid #dee2e6; border-left:2px solid #dee2e6; background:#fff;")
    th_ampm = ("padding:2px 3px; font-size:10px; font-weight:600; text-align:center; "
               "border-bottom:1px solid #dee2e6; color:#555; background:#fff;")
    th_sub  = ("padding:2px 2px; font-size:9px; text-align:center; width:18px; "
               "border-bottom:2px solid #dee2e6; color:#888; background:#fff;")
    th_sub_left = th_sub + " border-left:2px solid #dee2e6;"

    empty = ui.tags.th("", style=th_base)
    row1 = [
        ui.tags.th("Name", style=th_base + " width:94px;"),
        ui.tags.th("Role", style=th_base + " width:102px;"),
        ui.tags.th("Max Hrs/Week", style=th_base + " width:80px; text-align:center;"),
    ]
    row2 = [empty, empty, empty]
    row3 = [empty, empty, empty]

    for day in open_days:
        row1.append(ui.tags.th(day, colspan="4", style=th_day))
        for st in SHIFT_TYPES:
            row2.append(ui.tags.th(st, colspan="2",
                style=th_ampm + (" border-left:1px solid #eee;" if st == "AM" else "")))
            row3.append(ui.tags.th("✔", style=th_sub_left))
            row3.append(ui.tags.th("★", style=th_sub))

    td_base = "padding:2px 2px; vertical-align:middle; border-bottom:1px solid #f2f2f2;"
    td_chk  = td_base + " text-align:center; width:18px; border-left:2px solid #dee2e6;"
    td_sel  = td_base + " text-align:center; width:34px;"

    data_rows = []
    for i in range(n):
        d = defaults[i] if i < len(defaults) else {}
        tds = [
            ui.tags.td(
                ui.input_text(f"t_name_{i}", None,
                              value=d.get("name", ""),
                              placeholder=f"Employee {i+1}", width="92px"),
                style=td_base
            ),
            ui.tags.td(
                ui.input_select(f"t_role_{i}", None, choices=ROLES,
                                selected=d.get("role", ROLES[0]), width="100px"),
                style=td_base
            ),
            ui.tags.td(
                ui.input_numeric(f"t_maxh_{i}", None,
                                 value=d.get("max_hours", 40), min=1, max=80, width="38px"),
                style=td_base + " text-align:center;"
            ),
        ]
        for day in open_days:
            for st in SHIFT_TYPES:
                sid = f"{day}_{st}"
                is_avail = d.get(f"avail_{sid}", 1)
                pref_val = d.get(f"pref_{sid}", "3")
                tds.append(ui.tags.td(
                    ui.tags.input(
                        id=f"t_avail_{i}_{sid}",
                        name=f"t_avail_{i}_{sid}",
                        type="checkbox",
                        **{"checked": ""} if is_avail else {},
                        style="width:15px; height:15px; cursor:pointer; margin:0;"
                    ),
                    style=td_chk
                ))
                tds.append(ui.tags.td(
                    ui.tags.select(
                        *[ui.tags.option(
                            str(v), value=str(v),
                            **{"selected": ""} if str(v) == str(pref_val) else {}
                          ) for v in range(1, 6)],
                        id=f"t_pref_{i}_{sid}",
                        name=f"t_pref_{i}_{sid}",
                        style="width:34px; font-size:11px; padding:1px 0; border:1px solid #ced4da; border-radius:3px;"
                    ),
                    style=td_sel
                ))
        data_rows.append(ui.tags.tr(*tds))

    return ui.div(
        ui.tags.table(
            ui.tags.thead(
                ui.tags.tr(*row1),
                ui.tags.tr(*row2),
                ui.tags.tr(*row3),
            ),
            ui.tags.tbody(*data_rows),
            style="border-collapse:collapse; width:max-content;"
        ),
        style="overflow-x:auto; padding-bottom:8px;"
    )


# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────

app_ui = ui.page_fluid(
    ui.tags.style("""
        body { font-family: 'Segoe UI', sans-serif; background: #f8f9fa; }
        .metric-box {
            background:#fff; border:1px solid #dee2e6; border-radius:8px;
            padding:12px 16px; margin:0 10px 10px 0;
            display:inline-block; min-width:170px; vertical-align:top;
        }
        .metric-label { font-size:11px; color:#6c757d; margin-bottom:2px; text-transform:uppercase; }
        .metric-value { font-size:22px; font-weight:700; color:#212529; }
        .section-title { font-size:15px; font-weight:600; margin:22px 0 8px; color:#343a40;
                         border-bottom:2px solid #dee2e6; padding-bottom:4px; }
        .alert-box { padding:11px 15px; border-radius:6px; margin-bottom:14px; font-size:14px; }
        .alert-danger  { background:#f8d7da; color:#842029; border:1px solid #f5c2c7; }
        .alert-success { background:#d1e7dd; color:#0f5132; border:1px solid #badbcc; }
        .alert-info    { background:#cff4fc; color:#055160; border:1px solid #b6effb; }
        .change-log {
            font-family:monospace; background:#f1f3f5; padding:12px; border-radius:6px;
            font-size:13px; white-space:pre-wrap; margin-bottom:16px; border:1px solid #dee2e6;
        }
        h2 { margin-bottom:4px; }
        .form-label { margin-bottom:0 !important; }
        .shiny-input-container { margin-bottom:0 !important; }
        /* Compact inputs inside table cells */
        td input[type=number] { padding:1px 2px !important; font-size:12px !important; }
        td input[type=text]   { padding:2px 4px !important; font-size:12px !important; }
        td select             { padding:1px 1px !important; font-size:11px !important; }
        /* Remove the auto-generated label space from checkboxes in cells */
        td .shiny-input-container.shiny-input-checkbox { width:auto !important; }
        td .shiny-input-container.shiny-input-checkbox label { display:none !important; }
        td .shiny-input-container.shiny-input-checkbox input { margin:0 !important; }
        /* Style raw HTML checkboxes to match Shiny blue */
        td input[type=checkbox] {
            accent-color: #0d6efd;
            width:15px; height:15px; cursor:pointer;
        }
    """),

    ui.h2("🍽️ Restaurant Scheduling Optimizer"),
    ui.p("Constraint-based weekly scheduling with fairness and preference optimization.",
         style="color:#6c757d; margin-bottom:20px;"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h5("⚙️ Active Constraints"),
            ui.input_checkbox("availability", "Respect Availability",               True),
            ui.input_checkbox("max_hours",    "Enforce Max Weekly Hours",            True),
            ui.input_checkbox("no_clopening", "No PM → Next-Day AM (No Clopening)", True),
            ui.input_checkbox("fairness",     "Fairness: Balance Shift Counts",      True),
            ui.hr(),
            ui.input_action_button("run", "▶  Generate Schedule",
                                   class_="btn-primary w-100"),
            ui.hr(),
            ui.h5("🚨 Call-Out Simulation"),
            ui.input_select("emp_sel",   "Employee Calling Out", choices=[]),
            ui.input_select("shift_sel", "Their Affected Shift", choices=[]),
            ui.input_action_button("callout", "Mark Absent & Re-Optimize",
                                   class_="btn-danger w-100"),
            ui.hr(),
            ui.h6("Constraint Notes", style="color:#6c757d;"),
            ui.tags.small(
                ui.tags.b("Availability:"), " Only schedule available employees.", ui.tags.br(),
                ui.tags.b("Max Hours:"), " No employee exceeds their weekly cap.", ui.tags.br(),
                ui.tags.b("No Double Shift:"), " Max one shift per employee per day.", ui.tags.br(),
                ui.tags.b("No Clopening:"), " Blocks working a PM then next-day AM.", ui.tags.br(),
                ui.tags.b("Fairness:"), " Balances shift counts by role group.",
                style="color:#6c757d; line-height:2.0;"
            ),
            ui.hr(),
            ui.h6("Role Hierarchy", style="color:#6c757d;"),
            ui.tags.small(
                "Higher roles can fill lower-role slots:", ui.tags.br(),
                ui.tags.b("Manager"), " → can fill any slot", ui.tags.br(),
                ui.tags.b("Lead Server"), " → Lead, Server, or Host slots", ui.tags.br(),
                ui.tags.b("Server"), " → Server slots only", ui.tags.br(),
                ui.tags.b("Host"), " → Host slots only",
                style="color:#6c757d; line-height:2.0;"
            ),
            ui.hr(),
            ui.h5("🤖 AI Assistant"),
            ui.tags.small("Ask questions about the schedule below.",
                          style="color:#6c757d;"),
            width=290,
        ),

        ui.output_ui("status_banner"),
        ui.output_ui("callout_banner"),
        ui.output_ui("metrics_panel"),

        ui.navset_tab(

            # TAB 1 — Employee template
            ui.nav_panel(
                "👥 Employees",
                ui.div(style="height:14px;"),
                ui.div(
                    ui.input_numeric("n_employees", "Number of employees",
                                     value=6, min=1, max=MAX_EMPLOYEES, width="110px"),
                    ui.div(
                        ui.input_action_button("apply_n_emp", "Apply",
                                               class_="btn-outline-secondary btn-sm"),
                        style="margin-top:22px; margin-left:10px;"
                    ),
                    style="display:flex; align-items:flex-start;"
                ),
                ui.p("✔ = Available  |  ★ = Preference (1 low → 5 high)",
                     style="font-size:11px; color:#6c757d; margin:8px 0 4px;"),
                ui.hr(),
                ui.output_ui("employee_table"),
            ),

            # TAB 2 — Shift requirements
            ui.nav_panel(
                "📋 Shift Requirements",
                ui.div(style="height:14px;"),
                ui.p(
                    "Set staffing requirements for each shift. ",
                    ui.tags.b("Total must equal Mgr + Lead + Server + Host."),
                    style="font-size:13px; color:#6c757d;"
                ),
                ui.hr(),
                make_shift_form(),
            ),

            id="input_tabs",
        ),

        ui.div("📅 Weekly Schedule",  class_="section-title"),
        ui.output_data_frame("schedule"),

        ui.div("👥 Employee Summary", class_="section-title"),
        ui.output_data_frame("summary"),

        ui.div("🤖 AI Schedule Assistant", class_="section-title"),
        ui.p("Ask questions about the schedule — who is working, fairness, replacements, and more.",
             style="color:#6c757d; font-size:13px; margin-bottom:10px;"),
        ui.div(
            ui.output_ui("chat_history"),
            style="background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; "
                  "padding:12px; min-height:80px; max-height:360px; overflow-y:auto; "
                  "margin-bottom:10px; font-size:14px;"
        ),
        ui.div(
            ui.div(
                ui.input_text("chat_input", None,
                              placeholder="e.g. Who is working Monday AM?  Who has the fewest shifts?",
                              width="100%"),
                style="flex:1;"
            ),
            ui.div(
                ui.input_action_button("chat_send", "Ask",
                                       class_="btn-primary"),
                style="margin-left:8px;"
            ),
            style="display:flex; align-items:flex-start; gap:0;"
        ),
    )
)


# ─────────────────────────────────────────────
#  SERVER
# ─────────────────────────────────────────────

def server(input, output, session):

    # Pre-populate with example schedule so the app looks live on first load
    sched_store    = reactive.value(DEFAULT_SCHED_DF.copy())
    summ_store     = reactive.value(DEFAULT_SUMM_DF.copy())
    emp_reactive   = reactive.value(None)
    shift_reactive = reactive.value(None)
    error_msgs     = reactive.value([])
    change_log     = reactive.value("")
    metrics_store  = reactive.value(DEFAULT_METRICS.copy())
    callout_log    = reactive.value(set())
    n_emp_rows     = reactive.value(len(DEFAULT_EMP_DATA))

    # ── Employee table — entire table from one output_ui ─────────────
    @output
    @render.ui
    def employee_table():
        open_days = [d for d in ALL_DAYS if input[f"open_{d}"]()]
        return make_employee_table(n_emp_rows.get(), open_days, defaults=DEFAULT_EMP_DATA)

    @reactive.effect
    @reactive.event(input.apply_n_emp)
    def _apply_n():
        n_emp_rows.set(max(1, min(MAX_EMPLOYEES, int(input.n_employees()))))

    # ── Auto-computed shift totals (reactive sum of role inputs) ────────

    @output
    @render.ui
    def total_Mon_AM():
        total = (int(input["s_mgr_Mon_AM"]())  +
                 int(input["s_lead_Mon_AM"]()) +
                 int(input["s_srv_Mon_AM"]())  +
                 int(input["s_host_Mon_AM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Mon_PM():
        total = (int(input["s_mgr_Mon_PM"]())  +
                 int(input["s_lead_Mon_PM"]()) +
                 int(input["s_srv_Mon_PM"]())  +
                 int(input["s_host_Mon_PM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Tue_AM():
        total = (int(input["s_mgr_Tue_AM"]())  +
                 int(input["s_lead_Tue_AM"]()) +
                 int(input["s_srv_Tue_AM"]())  +
                 int(input["s_host_Tue_AM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Tue_PM():
        total = (int(input["s_mgr_Tue_PM"]())  +
                 int(input["s_lead_Tue_PM"]()) +
                 int(input["s_srv_Tue_PM"]())  +
                 int(input["s_host_Tue_PM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Wed_AM():
        total = (int(input["s_mgr_Wed_AM"]())  +
                 int(input["s_lead_Wed_AM"]()) +
                 int(input["s_srv_Wed_AM"]())  +
                 int(input["s_host_Wed_AM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Wed_PM():
        total = (int(input["s_mgr_Wed_PM"]())  +
                 int(input["s_lead_Wed_PM"]()) +
                 int(input["s_srv_Wed_PM"]())  +
                 int(input["s_host_Wed_PM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Thu_AM():
        total = (int(input["s_mgr_Thu_AM"]())  +
                 int(input["s_lead_Thu_AM"]()) +
                 int(input["s_srv_Thu_AM"]())  +
                 int(input["s_host_Thu_AM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Thu_PM():
        total = (int(input["s_mgr_Thu_PM"]())  +
                 int(input["s_lead_Thu_PM"]()) +
                 int(input["s_srv_Thu_PM"]())  +
                 int(input["s_host_Thu_PM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Fri_AM():
        total = (int(input["s_mgr_Fri_AM"]())  +
                 int(input["s_lead_Fri_AM"]()) +
                 int(input["s_srv_Fri_AM"]())  +
                 int(input["s_host_Fri_AM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Fri_PM():
        total = (int(input["s_mgr_Fri_PM"]())  +
                 int(input["s_lead_Fri_PM"]()) +
                 int(input["s_srv_Fri_PM"]())  +
                 int(input["s_host_Fri_PM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Sat_AM():
        total = (int(input["s_mgr_Sat_AM"]())  +
                 int(input["s_lead_Sat_AM"]()) +
                 int(input["s_srv_Sat_AM"]())  +
                 int(input["s_host_Sat_AM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Sat_PM():
        total = (int(input["s_mgr_Sat_PM"]())  +
                 int(input["s_lead_Sat_PM"]()) +
                 int(input["s_srv_Sat_PM"]())  +
                 int(input["s_host_Sat_PM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Sun_AM():
        total = (int(input["s_mgr_Sun_AM"]())  +
                 int(input["s_lead_Sun_AM"]()) +
                 int(input["s_srv_Sun_AM"]())  +
                 int(input["s_host_Sun_AM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    @output
    @render.ui
    def total_Sun_PM():
        total = (int(input["s_mgr_Sun_PM"]())  +
                 int(input["s_lead_Sun_PM"]()) +
                 int(input["s_srv_Sun_PM"]())  +
                 int(input["s_host_Sun_PM"]()))
        color = "#0f5132" if total > 0 else "#6c757d"
        return ui.HTML(
            f'<span style="font-size:16px;font-weight:700;color:{color}">{total}</span>'
        )

    # Reactively show/hide shift rows and update label when open day toggles
    @reactive.effect
    def _update_shift_visibility():
        js_parts = []
        for d in ALL_DAYS:
            is_open = input[f"open_{d}"]()
            disp  = "block" if is_open else "none"
            label = "— Open" if is_open else "— Closed"
            js_parts.append(
                f'var r=document.getElementById("shift_rows_{d}"); if(r) r.style.display="{disp}";'                f'var l=document.getElementById("open_label_{d}"); if(l) l.textContent="{label}";'            )
        ui.insert_ui(
            ui.tags.script("; ".join(js_parts)),
            selector="body", where="afterBegin", immediate=True
        )

        # ── Core optimizer ───────────────────────────────────────────────
    def run_optimization(emp_df, shift_df):
        constraints = {
            "availability": input.availability(),
            "max_hours":    input.max_hours(),
            "no_clopening": input.no_clopening(),
            "fairness":     input.fairness(),
        }
        sched, summ, errs = build_schedule(emp_df, shift_df, constraints)
        sched_store.set(sched)
        summ_store.set(summ)
        error_msgs.set(errs)
        metrics_store.set(
            compute_metrics(sched, summ, clean_shift(shift_df)) if not sched.empty else {}
        )
        ec = clean_emp(emp_df)
        sc = clean_shift(shift_df)
        if "Name"     in ec.columns: ui.update_select("emp_sel",   choices=ec["Name"].tolist())
        if "Shift_ID" in sc.columns: ui.update_select("shift_sel", choices=sc["Shift_ID"].tolist())

    # ── Generate schedule ────────────────────────────────────────────
    def get_open_days():
        return [d for d in ALL_DAYS if input[f"open_{d}"]()]

    @reactive.effect
    @reactive.event(input.run)
    def handle_run():
        open_days = get_open_days()
        if not open_days:
            error_msgs.set(["Select at least one open day before generating a schedule."])
            return
        emp = build_emp_df_from_inputs(input, n_emp_rows.get(), open_days)
        if emp is None or emp.empty:
            error_msgs.set(["Add at least one employee name in the Employees tab."])
            return
        shift = build_shift_df_from_inputs(input, open_days)
        emp_reactive.set(emp)
        shift_reactive.set(shift)
        change_log.set("")
        callout_log.set(set())
        run_optimization(emp, shift)

    # ── Call-out ─────────────────────────────────────────────────────
    @reactive.effect
    @reactive.event(input.callout)
    def handle_callout():
        emp  = emp_reactive.get()
        shift = shift_reactive.get()
        curr  = sched_store.get()

        if emp is None or shift is None or curr.empty:
            error_msgs.set(["Generate a schedule first before simulating a call-out."])
            return

        absent_emp     = input.emp_sel()
        affected_shift = input.shift_sel()
        constraints    = {
            "availability": input.availability(),
            "max_hours":    input.max_hours(),
            "no_clopening": input.no_clopening(),
            "fairness":     input.fairness(),
        }

        new_sched, new_summ, errs = resolve_callout(
            emp_df=emp, shift_df=shift, current_sched_df=curr,
            absent_emp=absent_emp, affected_shift_id=affected_shift,
            constraints=constraints, blocked_pairs=callout_log.get(),
        )

        if errs:
            error_msgs.set(errs)
            return

        callout_log.set(callout_log.get() | {(absent_emp, affected_shift)})

        old_map = dict(zip(curr["Shift"],      curr["Workers"]))
        new_map = dict(zip(new_sched["Shift"], new_sched["Workers"]))
        o = _parse_worker_names(old_map.get(affected_shift, ""))
        n = _parse_worker_names(new_map.get(affected_shift, ""))
        removed, added = o - n, n - o

        if removed or added:
            parts = []
            if removed: parts.append(f"  Removed: {', '.join(sorted(removed))}")
            if added:   parts.append(f"  Added:   {', '.join(sorted(added))}")
            log = (f"📋 Call-out change for {affected_shift}:\n" +
                   "\n".join(parts) + "\n\n✅ All other shifts remain unchanged.")
        else:
            log = f"✅ {affected_shift}: existing coverage sufficient — no swap needed."

        change_log.set(log)
        error_msgs.set([])
        sched_store.set(new_sched)
        summ_store.set(new_summ)
        metrics_store.set(
            compute_metrics(new_sched, new_summ, clean_shift(shift)) if not new_sched.empty else {}
        )

    # ── Outputs ──────────────────────────────────────────────────────
    @output
    @render.ui
    def status_banner():
        errs = error_msgs.get()
        if errs:
            body = "".join(f"<div>❌ {e}</div>" for e in errs)
            return ui.HTML(f'<div class="alert-box alert-danger">{body}</div>')
        if not sched_store.get().empty:
            n = len(sched_store.get())
            return ui.HTML(
                f'<div class="alert-box alert-success">'
                f'✅ Schedule generated — {n} shifts fully staffed.</div>'
            )
        return ui.HTML(
            '<div class="alert-box alert-info">'
            '📋 Example schedule shown below. Fill in the tabs and click Generate Schedule to create your own.</div>'
        )

    @output
    @render.ui
    def callout_banner():
        log = change_log.get()
        if not log:
            return ui.HTML("")
        return ui.HTML(f'<div class="change-log">{log}</div>')

    @output
    @render.ui
    def metrics_panel():
        m = metrics_store.get()
        if not m:
            return ui.HTML("")
        boxes = "".join(
            f'<div class="metric-box">'
            f'<div class="metric-label">{k}</div>'
            f'<div class="metric-value">{v}</div>'
            f'</div>'
            for k, v in m.items()
        )
        return ui.HTML(
            '<div class="section-title">📊 Schedule Metrics</div>'
            f'<div style="margin-bottom:20px;">{boxes}</div>'
        )

    @output
    @render.data_frame
    def schedule():
        df = sched_store.get()
        if df.empty:
            return df
        return render.DataGrid(df, width="100%", height="420px")

    @output
    @render.data_frame
    def summary():
        df = summ_store.get()
        if df.empty:
            return df
        cols = [c for c in df.columns if c not in ("Pref_Score", "Max_Pref_Score")]
        return render.DataGrid(df[cols], width="100%", height="420px")


    # ── AI Chat ──────────────────────────────────────────────────────

    chat_messages = reactive.value([])   # list of {"role": "user"|"assistant", "text": str}

    @reactive.effect
    @reactive.event(input.chat_send)
    def handle_chat():
        question = input.chat_input().strip()
        if not question:
            return

        # Immediately show the user message
        msgs = chat_messages.get() + [{"role": "user", "text": question}]
        chat_messages.set(msgs)
        ui.update_text("chat_input", value="")

        # Call AI and append response
        response = ask_schedule_ai(
            user_question   = question,
            sched_df        = sched_store.get(),
            summ_df         = summ_store.get(),
            metrics         = metrics_store.get(),
            change_log_text = change_log.get(),
        )
        chat_messages.set(chat_messages.get() + [{"role": "assistant", "text": response}])

    @output
    @render.ui
    def chat_history():
        msgs = chat_messages.get()
        if not msgs:
            return ui.HTML(
                '<span style="color:#adb5bd;">No messages yet. '
                'Generate a schedule then ask a question.</span>'
            )
        parts = []
        for m in msgs:
            if m["role"] == "user":
                parts.append(
                    f'<div style="margin-bottom:8px;">'
                    f'<span style="font-weight:600; color:#0d6efd;">You:</span> '
                    f'<span>{m["text"]}</span></div>'
                )
            else:
                parts.append(
                    f'<div style="margin-bottom:12px; padding:8px 10px; '
                    f'background:#fff; border-radius:6px; border:1px solid #dee2e6;">'
                    f'<span style="font-weight:600; color:#198754;">Assistant:</span><br>'
                    f'<span style="white-space:pre-wrap;">{m["text"]}</span></div>'
                )
        return ui.HTML("".join(parts))


app = App(app_ui, server)