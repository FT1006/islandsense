"""Streamlit UI for IslandSense MVP.

Single-page app showing:
- Fresh/Fuel JDI tiles
- Recommended actions
- Drill-down sailings table
- What-if sliders

To be implemented in M4.
"""

import streamlit as st


def main():
    """Main Streamlit app entry point."""
    st.set_page_config(page_title="IslandSense MVP", page_icon="🌊", layout="wide")

    st.title("🌊 IslandSense MVP")
    st.markdown(
        "Per-sailing disruption prediction and 72-hour early warning for Jersey supply chain"
    )

    st.info("🚧 Coming soon in M4...")

    st.markdown("""
    **Planned features:**
    - **Fresh/Fuel JDI tiles** with risk bands (Green/Amber/Red)
    - **Recommended actions** per category
    - **Sailing drill-down table** with p_sail and "Why" explanations
    - **What-if sliders** for wind/wave scenarios
    - **Export** functionality for ops briefs
    """)


if __name__ == "__main__":
    main()
