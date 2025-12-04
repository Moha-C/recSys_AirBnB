# src/ui/app_streamlit.py
import uuid

import pandas as pd
import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="The Right Trip Demo", layout="wide")

st.title("🏙️ The Right Trip – Context-Aware Airbnb Recommender")

# session id unique pour logging
if "session_id" not in st.session_state:
    st.session_state["session_id"] = str(uuid.uuid4())

st.sidebar.header("Search Context")
city_options = ["Paris", "Lyon", "Bordeaux"]
city = st.sidebar.selectbox("City", options=["Any"] + city_options, index=1)
n_guests = st.sidebar.slider("Number of guests", 1, 10, 2)
budget_min = st.sidebar.number_input("Min budget (per night)", min_value=0, value=50)
budget_max = st.sidebar.number_input("Max budget (per night)", min_value=0, value=200)
k = st.sidebar.slider("Top-K", min_value=5, max_value=50, value=10, step=5)

user_id = st.text_input("User ID (optional, for logging)", value="user_demo")
query = st.text_input(
    "What are you looking for?",
    value="cozy bright studio near park",
    help="Type your travel intent (style, vibe, location...).",
)


def log_event(user_id: str, item_id: int, action_type: str):
    payload = {
        "user_id": user_id,
        "item_id": item_id,
        "action_type": action_type,
        "session_id": st.session_state["session_id"],
    }
    try:
        requests.post(f"{API_URL}/log_interaction", json=payload, timeout=10)
    except Exception:
        pass


if st.button("Get recommendations"):
    if not query.strip():
        st.error("Please enter a non-empty query.")
    else:
        params = {
            "query": query,
            "k": k,
            "n_guests": int(n_guests),
            "budget_min": float(budget_min),
            "budget_max": float(budget_max),
        }

        if city != "Any":
            params["city"] = city

        if user_id:
            params["user_id"] = user_id

        try:
            r = requests.get(f"{API_URL}/recommend", params=params, timeout=60)
            if r.status_code != 200:
                st.error(f"API error {r.status_code}: {r.text}")
            else:
                data = r.json()
                df = pd.DataFrame(data["results"])
                st.subheader("Recommended listings")
                if df.empty:
                    st.info("No results.")
                else:
                    for i, row in df.iterrows():
                        price_val = row.get("price", None)
                        price_str = (
                            f"{price_val:.0f} €"
                            if isinstance(price_val, (int, float)) and pd.notna(price_val)
                            else "N/A"
                        )

                        title = f"#{i+1} – {row['name']}"
                        with st.expander(title):
                            cols = st.columns([1, 2])
                            with cols[0]:
                                pic_url = row.get("picture_url", "")
                                if isinstance(pic_url, str) and pic_url.strip():
                                    st.image(pic_url, use_container_width=True)


                            with cols[1]:
                                st.write(
                                    f"**Neighbourhood:** {row['neighbourhood']} ({row['city']})"
                                )
                                st.write(
                                    f"**Price:** {price_str}  |  **Accommodates:** {row['accommodates']}"
                                )
                                st.write(f"**Score:** {row['score']:.3f}")

                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("👍 Like", key=f"like_{row['listing_id']}"):
                                        log_event(
                                            user_id=user_id,
                                            item_id=int(row["listing_id"]),
                                            action_type="thumb_up",
                                        )
                                        st.success("Logged 👍")
                                with col2:
                                    if st.button(
                                        "👎 Not interested",
                                        key=f"dislike_{row['listing_id']}",
                                    ):
                                        log_event(
                                            user_id=user_id,
                                            item_id=int(row["listing_id"]),
                                            action_type="thumb_down",
                                        )
                                        st.success("Logged 👎")
        except Exception as e:
            st.error(f"Error contacting API: {e}")
