"use client";

import React, { useState } from "react";
import {
  Search,
  MapPin,
  Users,
  DollarSign,
  ThumbsUp,
  ThumbsDown,
  Loader2,
  LogOut,
  User as UserIcon,
} from "lucide-react";

const API_URL = "http://localhost:8000";

export default function TripRecommender() {
  const [sessionId] = useState(() =>
    Math.random().toString(36).substring(7)
  );

  // ---- Auth / login ----
  const [loginUserId, setLoginUserId] = useState("alice");
  const [loginPassword, setLoginPassword] = useState("");
  const [authUser, setAuthUser] = useState(null); // {user_id, display_name}
  const [authLoading, setAuthLoading] = useState(false);
  const [authError, setAuthError] = useState(null);

  // ---- Search state ----
  const [query, setQuery] = useState("cozy bright studio near park");
  const [city, setCity] = useState("Paris");
  const [nGuests, setNGuests] = useState(2);
  const [budgetMin, setBudgetMin] = useState(50);
  const [budgetMax, setBudgetMax] = useState(200);
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // ----------------------------------------------------------
  // Login / logout
  // ----------------------------------------------------------
  const handleLogin = async () => {
    setAuthError(null);
    if (!loginUserId.trim() || !loginPassword.trim()) {
      setAuthError("Please enter user ID and password.");
      return;
    }

    setAuthLoading(true);
    try {
      const response = await fetch(`${API_URL}/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: loginUserId,
          password: loginPassword,
        }),
      });

      if (!response.ok) {
        throw new Error("Invalid credentials");
      }
      const data = await response.json();
      // data = { user_id, display_name }
      setAuthUser(data);
      setLoginPassword("");
    } catch (err) {
      console.error(err);
      setAuthError(err.message || "Login failed");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleLogout = () => {
    setAuthUser(null);
    setResults([]);
    setQuery("cozy bright studio near park");
    setCity("Paris");
    setNGuests(2);
    setBudgetMin(50);
    setBudgetMax(200);
    setTopK(10);
  };

  // ----------------------------------------------------------
  // Logging interactions
  // ----------------------------------------------------------
  const logEvent = async (itemId, actionType) => {
    try {
      await fetch(`${API_URL}/log_interaction`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: authUser?.user_id || null,
          item_id: itemId,
          action_type: actionType,
          session_id: sessionId,
          metadata: {
            query,
            city,
            n_guests: nGuests,
            budget_min: budgetMin,
            budget_max: budgetMax,
          },
        }),
      });
    } catch (err) {
      console.error("Failed to log interaction:", err);
    }
  };

  // ----------------------------------------------------------
  // Search / recommendations
  // ----------------------------------------------------------
  const handleSearch = async () => {
    if (!authUser) {
      setError("You must be logged in.");
      return;
    }
    if (!query.trim()) {
      setError("Please enter a search query.");
      return;
    }

    setLoading(true);
    setError(null);

    const params = new URLSearchParams({
      query,
      k: topK,
      n_guests: nGuests,
      budget_min: budgetMin,
      budget_max: budgetMax,
      user_id: authUser.user_id,
    });

    if (city !== "Any") {
      params.append("city", city);
    }

    try {
      const response = await fetch(`${API_URL}/recommend?${params.toString()}`);
      if (!response.ok) throw new Error(`API error: ${response.status}`);

      const data = await response.json();
      setResults(data.results || []);
    } catch (err) {
      console.error(err);
      setError(err.message || "Failed to fetch recommendations.");
    } finally {
      setLoading(false);
    }
  };

  const handleLike = (listingId) => {
    logEvent(listingId, "thumb_up");
  };

  const handleDislike = (listingId) => {
    logEvent(listingId, "thumb_down");
    // Optionnel : filtrer localement
    setResults((prev) => prev.filter((r) => r.listing_id !== listingId));
  };

  // ----------------------------------------------------------
  // UI – Login screen
  // ----------------------------------------------------------
  if (!authUser) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 flex items-center justify-center">
        <div className="bg-white rounded-2xl shadow-lg border border-slate-200 p-8 w-full max-w-md">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-10 h-10 bg-blue-500 rounded-xl flex items-center justify-center">
              <MapPin className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-slate-900">
                The Right Trip
              </h1>
              <p className="text-xs text-slate-500">
                Personalized Airbnb-style recommendations
              </p>
            </div>
          </div>

          <h2 className="text-lg font-semibold text-slate-900 mb-4">
            Log in to your demo profile
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Users disponibles par défaut :
            <br />
            Elias / elias123 – budget city explorer
            <br />
            Mohammed / mohammed123 – budget traveller
            <br />
            Wacim / waciml123 – design lover
            <br />
            Samuel / samuel123 – family trips
            <br />
            Sofiane / sofiane123 – luxury trips
          </p>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                User ID
              </label>
              <input
                type="text"
                value={loginUserId}
                onChange={(e) => setLoginUserId(e.target.value)}
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="alice"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Password
              </label>
              <input
                type="password"
                value={loginPassword}
                onChange={(e) => setLoginPassword(e.target.value)}
                className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                placeholder="******"
              />
            </div>

            {authError && (
              <p className="text-xs text-red-500 mt-1">{authError}</p>
            )}

            <button
              onClick={handleLogin}
              disabled={authLoading}
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 text-white text-sm font-medium py-2.5 hover:bg-blue-700 disabled:opacity-60"
            >
              {authLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Logging in...
                </>
              ) : (
                "Log in"
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ----------------------------------------------------------
  // UI – main app (once logged in)
  // ----------------------------------------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header */}
      <div className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-500 rounded-xl flex items-center justify-center">
              <MapPin className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-semibold text-slate-900">
                The Right Trip
              </h1>
              <p className="text-xs text-slate-500">
                Multimodal, context-aware recommendations
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 text-sm text-slate-700">
              <UserIcon className="w-4 h-4 text-slate-500" />
              <div className="flex flex-col">
                <span className="font-medium">{authUser.display_name}</span>
                <span className="text-xs text-slate-500">
                  id: {authUser.user_id}
                </span>
              </div>
            </div>
            <button
              onClick={handleLogout}
              className="inline-flex items-center gap-1 rounded-xl border border-slate-300 px-3 py-1.5 text-xs text-slate-700 hover:bg-slate-100"
            >
              <LogOut className="w-4 h-4" />
              Logout
            </button>
          </div>
        </div>
      </div>

      {/* Main layout */}
      <div className="max-w-7xl mx-auto px-6 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Filters Sidebar */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6 sticky top-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-6">
                Search Filters
              </h2>

              {/* Query */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Trip description
                </label>
                <div className="relative">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    className="w-full rounded-xl border border-slate-300 pl-10 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    placeholder="cozy bright studio near park"
                  />
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
                </div>
              </div>

              {/* City */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  City
                </label>
                <select
                  value={city}
                  onChange={(e) => setCity(e.target.value)}
                  className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                >
                  <option value="Any">Any</option>
                  <option value="Paris">Paris</option>
                  <option value="Lyon">Lyon</option>
                  <option value="Bordeaux">Bordeaux</option>
                </select>
              </div>

              {/* Guests */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Guests
                </label>
                <div className="flex items-center gap-2">
                  <Users className="w-4 h-4 text-slate-400" />
                  <input
                    type="number"
                    min={1}
                    max={10}
                    value={nGuests}
                    onChange={(e) =>
                      setNGuests(parseInt(e.target.value || "1", 10))
                    }
                    className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>
              </div>

              {/* Budget */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Budget per night (€)
                </label>
                <div className="flex gap-2">
                  <div className="flex-1 flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-slate-400" />
                    <input
                      type="number"
                      value={budgetMin}
                      onChange={(e) =>
                        setBudgetMin(parseInt(e.target.value || "0", 10))
                      }
                      className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                  <span className="text-slate-400 self-center text-xs">to</span>
                  <div className="flex-1">
                    <input
                      type="number"
                      value={budgetMax}
                      onChange={(e) =>
                        setBudgetMax(parseInt(e.target.value || "0", 10))
                      }
                      className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                    />
                  </div>
                </div>
              </div>

              {/* Top-K */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-slate-700 mb-2">
                  Number of results (Top-K)
                </label>
                <input
                  type="number"
                  min={5}
                  max={30}
                  step={1}
                  value={topK}
                  onChange={(e) =>
                    setTopK(parseInt(e.target.value || "10", 10))
                  }
                  className="w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Search button */}
              <button
                onClick={handleSearch}
                disabled={loading}
                className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-blue-600 text-white text-sm font-medium py-2.5 hover:bg-blue-700 disabled:opacity-60"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Searching...
                  </>
                ) : (
                  <>
                    <Search className="w-4 h-4" />
                    Get recommendations
                  </>
                )}
              </button>

              {error && (
                <p className="mt-3 text-xs text-red-500">
                  {error}
                </p>
              )}
            </div>
          </div>

          {/* Results */}
          <div className="lg:col-span-3">
            <h2 className="text-lg font-semibold text-slate-900 mb-4">
              Recommended listings
            </h2>

            {results.length === 0 && !loading && (
              <p className="text-sm text-slate-500">
                No results yet. Start by describing your ideal stay and hit
                &quot;Get recommendations&quot;.
              </p>
            )}

            <div className="space-y-4">
              {results.map((listing, idx) => {
                const price =
                  typeof listing.price === "number" &&
                  !Number.isNaN(listing.price)
                    ? `${listing.price.toFixed(0)} €`
                    : "N/A";

                return (
                  <div
                    key={listing.listing_id}
                    className="bg-white rounded-2xl shadow-sm border border-slate-200 overflow-hidden"
                  >
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-0">
                      {/* Image */}
                      <div className="md:col-span-1 relative">
                        {listing.picture_url ? (
                          <img
                            src={listing.picture_url}
                            alt={listing.name}
                            className="w-full h-56 object-cover"
                          />
                        ) : (
                          <div className="w-full h-56 bg-slate-100 flex items-center justify-center text-xs text-slate-400">
                            No image available
                          </div>
                        )}
                      </div>

                      {/* Text */}
                      <div className="md:col-span-2 p-4 flex flex-col justify-between">
                        <div>
                          <p className="text-xs text-slate-400 mb-1">
                            #{idx + 1} – Score {listing.score.toFixed(3)}
                          </p>
                          <h3 className="text-base font-semibold text-slate-900 mb-1">
                            {listing.name}
                          </h3>
                          <p className="text-xs text-slate-500 mb-2">
                            {listing.neighbourhood} ({listing.city})
                          </p>
                          <div className="flex items-center gap-4 text-xs text-slate-600">
                            <span className="inline-flex items-center gap-1">
                              <Users className="w-3 h-3" />
                              Sleeps {listing.accommodates}
                            </span>
                            <span className="inline-flex items-center gap-1">
                              <DollarSign className="w-3 h-3" />
                              {price} / night
                            </span>
                          </div>
                        </div>

                        <div className="flex gap-2 mt-3">
                          <button
                            onClick={() =>
                              handleLike(listing.listing_id)
                            }
                            className="inline-flex items-center gap-1 rounded-xl border border-emerald-500 text-emerald-600 text-xs px-3 py-1.5 hover:bg-emerald-50"
                          >
                            <ThumbsUp className="w-3 h-3" />
                            Like
                          </button>
                          <button
                            onClick={() =>
                              handleDislike(listing.listing_id)
                            }
                            className="inline-flex items-center gap-1 rounded-xl border border-rose-500 text-rose-600 text-xs px-3 py-1.5 hover:bg-rose-50"
                          >
                            <ThumbsDown className="w-3 h-3" />
                            Not interested
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
