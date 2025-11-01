import streamlit as st
import pandas as pd
from schedule_ga import genetic_algorithm, load_data

st.set_page_config(page_title="GA TV Scheduling - 3 Trials", page_icon="📺", layout="centered")

st.title("📺 Genetic Algorithm TV Scheduling - 3 Trials")
st.markdown("#### Developed by Milashini Saravanan (JIE42903 - Evolutionary Computing)")

# === Load Dataset ===
try:
    data = load_data("data/program_ratings.csv")
    st.subheader("📊 Dataset Preview")
    st.dataframe(data)
except Exception as e:
    st.error(f"⚠️ Error loading CSV file: {e}")
    st.stop()

st.divider()
st.markdown("### ⚙️ Trial Parameter Settings")

# Define 3 trial parameter sets
trial_params = [
    {"trial": 1, "co_rate": 0.8, "mut_rate": 0.02},
    {"trial": 2, "co_rate": 0.6, "mut_rate": 0.04},
    {"trial": 3, "co_rate": 0.9, "mut_rate": 0.01}
]

# Allow user to modify trials if they want
for t in trial_params:
    col1, col2 = st.columns(2)
    with col1:
        t["co_rate"] = st.number_input(
            f"Trial {t['trial']} - Crossover Rate (CO_R)",
            0.0, 0.95, t["co_rate"], 0.01
        )
    with col2:
        t["mut_rate"] = st.number_input(
            f"Trial {t['trial']} - Mutation Rate (MUT_R)",
            0.01, 0.05, t["mut_rate"], 0.01
        )

st.divider()

# === Run all trials ===
if st.button("🚀 Run All 3 Trials"):
    results = []
    for t in trial_params:
        schedule, score = genetic_algorithm(data, t["co_rate"], t["mut_rate"])
        results.append({
            "Trial": t["trial"],
            "CO_R": t["co_rate"],
            "MUT_R": t["mut_rate"],
            "Fitness": score,
            "Schedule": schedule
        })

    st.success("✅ All trials completed successfully!")
    st.divider()

    # === Display results ===
    for r in results:
        st.markdown(f"### 🧩 Trial {r['Trial']} Results")
        st.markdown(f"- **Crossover Rate (CO_R):** {r['CO_R']}")
        st.markdown(f"- **Mutation Rate (MUT_R):** {r['MUT_R']}")
        st.markdown(f"- **Fitness Score:** `{r['Fitness']}`")

        df = pd.DataFrame({
            "Order": range(1, len(r["Schedule"]) + 1),
            "Program": r["Schedule"]
        })
        st.table(df)
        st.divider()

    # === Summary Table ===
    summary_df = pd.DataFrame(
        [{
            "Trial": r["Trial"],
            "Crossover Rate (CO_R)": r["CO_R"],
            "Mutation Rate (MUT_R)": r["MUT_R"],
            "Fitness Score": r["Fitness"]
        } for r in results]
    )
    st.markdown("### 📊 Summary of All Trials")
    st.table(summary_df)

    # Optional: Best trial highlight
    best_trial = max(results, key=lambda x: x["Fitness"])
    st.success(
        f"🏆 Best Performance: Trial {best_trial['Trial']} "
        f"(CO_R = {best_trial['CO_R']}, MUT_R = {best_trial['MUT_R']}) "
        f"→ Fitness Score = {best_trial['Fitness']}"
    )
