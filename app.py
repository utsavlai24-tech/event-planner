%%writefile app.py
import streamlit as st
import pandas as pd
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
import jason
import streamlit as st

# =========================================
# GEMINI CONFIG
# =========================================
MODEL = "gemini-2.5-flash"

base_llm = ChatGoogleGenerativeAI(
    model=MODEL,
    google_api_key=st.secrects.get("GEMINI_API_KEY"),
    temperature=0.25
)

# =========================================
# RICH STRUCTURED SCHEMAS (Clean Data, High Detail)
# =========================================
class VenueSchema(BaseModel):
    venue_name_idea: str = Field(description="A catchy, realistic premium venue name or style recommendation matching the city and theme")
    venue_type: str = Field(description="Detailed setting description (e.g., Luxury Outdoor Beachfront Lawn, Premium Heritage Banquet Hall)")
    seating_style: str = Field(description="Specific seating style and layout geometry description")
    decorations: List[str] = Field(description="List of exactly 3 descriptive decoration concepts matching the theme")
    logistics_note: str = Field(description="Short operational note on sound setup, parking, power backup, or structural staging needs")
    estimated_cost: int = Field(description="Estimated raw integer cost for venue rental and setup decoration package")

class CateringSchema(BaseModel):
    service_style: str = Field(description="Style of culinary experience (e.g., Live Interactive Stations, Multi-Course Sit-Down, Signature Premium Buffet)")
    starter: List[str] = Field(description="List of exactly 3 distinct high-end gourmet appetizer names")
    main_course: List[str] = Field(description="List of exactly 3 distinct rich signature main course offerings")
    desserts: List[str] = Field(description="List of exactly 2 distinct premium artisanal sweet creations")
    drinks: List[str] = Field(description="List of exactly 2 distinct curated signature beverages or mixology options")
    estimated_cost: int = Field(description="Estimated raw integer cost for full hospitality, catering management, and food supply per guest total cost")

class TimelineItem(BaseModel):
    time: str = Field(description="Exact time slot stamp (e.g., 06:30 PM - 07:00 PM)")
    phase: str = Field(description="Short phase title (e.g., Guest Arrival, Grand Entrance, Main Banquet, After Party)")
    activity: str = Field(description="Comprehensive summary of operational activities, entertainment highlights, or coordination tasks during this phase")

class ScheduleSchema(BaseModel):
    timeline: List[TimelineItem] = Field(description="Sequence log array containing exactly 4 chronological event phases")

# Native API structural bindings
venue_llm = base_llm.with_structured_output(VenueSchema)
catering_llm = base_llm.with_structured_output(CateringSchema)
schedule_llm = base_llm.with_structured_output(ScheduleSchema)

# =========================================
# STATE DEFINITION
# =========================================
class EventState(TypedDict):
    event_type: str
    location: str
    guests: int
    budget: int
    theme: str
    venue_plan: dict
    catering_plan: dict
    schedule_plan: dict
    final_plan: dict

# =========================================
# AGENT NODES
# =========================================
def venue_agent(state):
    prompt = f"Plan premium venue assets for a {state['event_type']} in {state['location']}. Guests: {state['guests']}, Budget: {state['budget']} INR, Design Theme: {state['theme']}. Generate crisp, highly descriptive plans."
    try:
        response = venue_llm.invoke(prompt)
        state["venue_plan"] = response.model_dump()
    except:
        state["venue_plan"] = {"venue_name_idea": "Grand Elite Plaza", "venue_type": "Premium Banquet Hall Hall", "seating_style": "Round-Table Cluster Arrangement", "decorations": ["Ambient Fairy Light Canopy", "Themed Floral Backdrops", "Luxury Velvet Drapes"], "logistics_note": "Ensure 15KW audio support and heavy generator backup.", "estimated_cost": int(state['budget'] * 0.45)}
    return state

def catering_agent(state):
    prompt = f"Plan a premium culinary blueprint for a {state['event_type']}. Theme: {state['theme']}, Guests: {state['guests']}. Ensure the menu matches the tone of the event beautifully."
    try:
        response = catering_llm.invoke(prompt)
        state["catering_plan"] = response.model_dump()
    except:
        state["catering_plan"] = {"service_style": "Signature Live Stations", "starter": ["Crisp Gourmet Sliders", "Glazed Paneer Skewers"], "main_course": ["Artisanal Saffron Rice", "Rich Slow-Cooked Curry"], "desserts": ["Warm Lava Cake", "Exotic Fruit Tarts"], "drinks": ["Curated Mint Coolers", "Infused Mocktails"], "estimated_cost": int(state['budget'] * 0.35)}
    return state

def schedule_agent(state):
    prompt = f"Create a comprehensive 4-phase master schedule itinerary timeline for a {state['event_type']} hosting {state['guests']} guests. Make it highly professional and logistically smooth."
    try:
        response = schedule_llm.invoke(prompt)
        state["schedule_plan"] = response.model_dump()
    except:
        state["schedule_plan"] = {"timeline": [{"time": "06:00 PM - 07:00 PM", "phase": "Welcome Reception", "activity": "Red carpet guest arrival with soft background acoustic music and pass-around welcome mocktails."}, {"time": "07:00 PM - 08:30 PM", "phase": "Main Formal Program", "activity": "Grand host entry introduction, key entertainment showcases, interactive segments, and thematic media presentations."}, {"time": "08:30 PM - 10:00 PM", "phase": "Culinary Banquet", "activity": "Premium live-counter dining floor opens with ambient jazz music tracks playing in the background."}, {"time": "10:00 PM - 11:00 PM", "phase": "Wrap-Up & Departure", "activity": "Photo ops on the main stage deck, distribution of party favors, and controlled breakdown checkout."}]}
    return state

def organizer_agent(state):
    v_price = state["venue_plan"].get("estimated_cost", 0)
    c_price = state["catering_plan"].get("estimated_cost", 0)

    state["final_plan"] = {
        "event_type": state["event_type"],
        "location": state["location"],
        "guests": state["guests"],
        "theme": state["theme"],
        "budget": state["budget"],
        "estimated_total": v_price + c_price,
        "venue": state["venue_plan"],
        "catering": state["catering_plan"],
        "schedule": state["schedule_plan"]
    }
    return state

# =========================================
# LANGGRAPH WORKFLOW
# =========================================
builder = StateGraph(EventState)
builder.add_node("Venue Agent", venue_agent)
builder.add_node("Catering Agent", catering_agent)
builder.add_node("Schedule Agent", schedule_agent)
builder.add_node("Organizer Agent", organizer_agent)

builder.set_entry_point("Venue Agent")
builder.add_edge("Venue Agent", "Catering Agent")
builder.add_edge("Catering Agent", "Schedule Agent")
builder.add_edge("Schedule Agent", "Organizer Agent")
builder.add_edge("Organizer Agent", END)

event_planner_app = builder.compile()

# =========================================
# STREAMLIT PREMIUM UI
# =========================================
st.set_page_config(page_title="AI Event Architect", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .main-title { font-size: 2.6rem; font-weight: 800; color: #1E293B; margin-bottom: 2px; }
    .sub-title { color: #64748B; font-size: 1.1rem; margin-bottom: 25px; }
    .section-header { font-size: 1.5rem; font-weight: 700; color: #0F172A; margin-top: 15px; margin-bottom: 15px; border-bottom: 2px solid #E2E8F0; padding-bottom: 8px; }
    .card-block { background: white; padding: 24px; border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.03); border: 1px solid #E2E8F0; margin-bottom: 20px; }
    .item-badge { background-color: #F1F5F9; color: #334155; padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 0.85rem; display: inline-block; margin: 4px; border: 1px solid #E2E8F0; }
    .timeline-node { border-left: 4px solid #4F46E5; padding-left: 20px; position: relative; margin-bottom: 25px; }
    .timeline-time { font-size: 0.95rem; font-weight: 700; color: #4F46E5; background: #EEF2F6; padding: 2px 10px; border-radius: 6px; }
    .timeline-phase { font-size: 1.15rem; font-weight: 700; color: #1E293B; margin-top: 5px; margin-bottom: 3px; }
    .timeline-text { font-size: 0.95rem; color: #475569; line-height: 1.5; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🎉 AI Event Architect Pro</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Multi-Agent Production Coordination Framework</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("📋 Technical Blueprint Parameters")
    event_type = st.selectbox("Event Blueprint Framework", ["Wedding Reception", "Corporate Gala", "Birthday Party", "Conference", "Product Launch"])
    location = st.text_input("📍 Deployment Target City", "Bangalore")
    guests = st.slider("👥 Guest Payload Volume", 10, 1000, 250, step=10)
    budget = st.number_input("💰 FinOps Capital Budget (INR)", min_value=5000, max_value=10000000, value=600000, step=25000)
    theme = st.text_input("✨ Creative Design Theme Focus", "Royal Traditional Elegance")

    st.markdown("---")
    submit = st.button("🚀 Coordinate Agent Consensus", type="primary", use_container_width=True)

if submit:
    with st.spinner("Processing framework nodes across agent matrix layers..."):
        results = event_planner_app.invoke({
            "event_type": event_type, "location": location, "guests": guests, "budget": budget, "theme": theme,
            "venue_plan": {}, "catering_plan": {}, "schedule_plan": {}, "final_plan": {}
        })

        final = results.get("final_plan", {})
        v_data = final.get("venue", {})
        c_data = final.get("catering", {})
        s_data = final.get("schedule", {})

        v_price = v_data.get("estimated_cost", 0)
        c_price = c_data.get("estimated_cost", 0)
        total_spent = final.get("estimated_total", v_price + c_price)
        margin_buffer = budget - total_spent

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Target Capital Allocation", f"₹{budget:,}")
        m2.metric("Consolidated Estimate", f"₹{total_spent:,}")
        m3.metric("Venue Asset Allocation", f"₹{v_price:,}")

        if margin_buffer >= 0:
            m4.metric("Capital Safety Margin", f"₹{margin_buffer:,}", delta=f"₹{margin_buffer:,} Under Base Budget")
        else:
            m4.metric("Budget Deficit Overhead", f"₹{abs(margin_buffer):,}", delta=f"-₹{abs(margin_buffer):,} Deficit Balance", delta_color="inverse")

        st.divider()

        col_main, col_side = st.columns([1.1, 0.9])

        with col_main:
            # --- ITINERARY ---
            st.markdown('<div class="section-header">⏳ Master Event Production Timeline Itinerary</div>', unsafe_allow_html=True)
            timeline = s_data.get("timeline", [])
            if timeline:
                for idx, slot in enumerate(timeline):
                    st.markdown(f"""
                    <div class="timeline-node">
                        <span class="timeline-time">⏱️ {slot.get('time', 'TBD')}</span>
                        <div class="timeline-phase">Phase {idx+1}: {slot.get('phase', 'Operational Stage')}</div>
                        <div class="timeline-text">{slot.get('activity', 'No actions loaded.')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Itinerary pipeline execution returned empty state data.")

            # --- VENUE AND LOGISTICS ---
            st.markdown('<div class="section-header">🏨 Venue Sourcing & Operational Logistics Setup</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card-block">', unsafe_allow_html=True)
                st.markdown(f"#### **🏛️ Proposed Venue Conception:** *\"{v_data.get('venue_name_idea', 'N/A')}\"*")
                st.markdown(f"**Structural Environment Matrix:** {v_data.get('venue_type', 'N/A')}")
                st.markdown(f"**Floor Arrangement Config:** `{v_data.get('seating_style', 'N/A')}`")

                st.markdown("<div style='margin-top:14px; font-weight:700;'>🎨 Creative Aesthetic Styling Elements:</div>", unsafe_allow_html=True)
                decors = v_data.get("decorations", [])
                for item in decors:
                    # FIX: Quotes inside the f-string changed to single quotes
                    st.markdown(f"<span class='item-badge'>✨ {item}</span>", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:16px; border-top:1px dashed #E2E8F0; padding-top:12px;'>", unsafe_allow_html=True)
                st.markdown(f"⚙️ **Logistical Operations Directive:** *{v_data.get('logistics_note', 'No explicit logistics recorded.')}*")
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

        with col_side:
            # --- FOOD DESIGN ---
            st.markdown('<div class="section-header">🍽️ Culinary Blueprint & Gastronomy Design</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="card-block">', unsafe_allow_html=True)
                st.markdown(f"#### **👨‍🍳 Catering Execution Style:** `{c_data.get('service_style', 'Standard Buffet')}`")
                st.markdown("---")

                food_categories = [
                    ("🍱 Premium Amuse-Bouche / Starters", "starter"),
                    ("🍲 Artisanal Mains / Entrées", "main_course"),
                    ("🍰 Curated Pastry Craft / Desserts", "desserts"),
                    ("🍹 Bespoke Mixology / Refreshments", "drinks")
                ]

                for label, key in food_categories:
                    st.markdown(f"<div style='font-weight:700; color:#334155; margin-top:8px;'>{label}</div>", unsafe_allow_html=True)
                    items = c_data.get(key, [])
                    if items:
                        for dish in items:
                            # FIX: Quotes inside the f-string changed to single quotes
                            st.markdown(f"<span class='item-badge'>✔️ {dish}</span>", unsafe_allow_html=True)
                    else:
                        st.caption("Standard allocation issue.")
                    st.markdown("<div style='margin-bottom:12px;'></div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- COST DISTRIBUTION CHART ---
            st.markdown('<div class="section-header">📊 Capital Cost Component Weight Chart</div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame({
                "Asset Group": ["Venue Operational Stake", "Hospitality & Culinary Sourcing"],
                "Capital Allocated (INR)": [v_price, c_price]
            })
            st.bar_chart(chart_df.set_index("Asset Group"), color="#4F46E5")
