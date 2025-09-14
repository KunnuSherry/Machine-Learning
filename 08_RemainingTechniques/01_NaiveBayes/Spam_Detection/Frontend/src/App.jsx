import React, { useState, useEffect } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://localhost:8000";

function App() {
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  const submit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });

      if (!res.ok) throw new Error("Server error");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const clearAll = () => {
    setMessage("");
    setResult(null);
    setError(null);
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };

    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  return (
    <div className="min-h-screen relative overflow-hidden bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-900 p-4">
      {/* Animated Background Elements */}
      <div className="absolute inset-0 overflow-hidden">
        <div 
          className="absolute w-96 h-96 bg-blue-500/10 rounded-full blur-3xl animate-pulse"
          style={{
            left: mousePosition.x * 0.1,
            top: mousePosition.y * 0.1,
            transform: 'translate(-50%, -50%)'
          }}
        />
        <div 
          className="absolute w-80 h-80 bg-purple-500/10 rounded-full blur-3xl animate-pulse delay-1000"
          style={{
            right: mousePosition.x * 0.05,
            bottom: mousePosition.y * 0.05,
            transform: 'translate(50%, 50%)'
          }}
        />
        <div 
          className="absolute w-64 h-64 bg-pink-500/10 rounded-full blur-3xl animate-pulse delay-2000"
          style={{
            left: mousePosition.x * 0.08,
            bottom: mousePosition.y * 0.08,
            transform: 'translate(-30%, 30%)'
          }}
        />
      </div>

      {/* Floating Particles */}
      <div className="absolute inset-0">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute w-2 h-2 bg-white/20 rounded-full animate-pulse"
            style={{
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              animationDelay: `${Math.random() * 3}s`,
              animationDuration: `${2 + Math.random() * 3}s`
            }}
          />
        ))}
      </div>

      <div className="relative z-10 flex items-center justify-center min-h-screen">
        <div className="w-full max-w-lg">
          {/* Main Card with 3D Effects */}
          <div 
            className="relative bg-white/10 backdrop-blur-xl rounded-3xl shadow-2xl p-8 space-y-8 border border-white/20"
            style={{
              transform: `perspective(1000px) rotateX(${(mousePosition.y - window.innerHeight/2) * 0.01}deg) rotateY(${(mousePosition.x - window.innerWidth/2) * 0.01}deg)`,
              transition: 'transform 0.1s ease-out'
            }}
          >
            {/* Glowing Border Effect */}
            <div className="absolute inset-0 rounded-3xl bg-gradient-to-r from-blue-500/20 via-purple-500/20 to-pink-500/20 blur-sm -z-10" />
            
            {/* Header with 3D Text */}
            <div className="text-center">
              <h1 
                className="text-4xl font-bold bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent mb-2"
                style={{
                  textShadow: '0 0 20px rgba(147, 51, 234, 0.5)',
                  transform: 'translateZ(20px)'
                }}
              >
                📨 SMS Spam Classifier
              </h1>
              <div className="w-24 h-1 bg-gradient-to-r from-blue-400 to-pink-400 mx-auto rounded-full animate-pulse" />
            </div>

            <form onSubmit={submit} className="space-y-6">
              {/* Input Section */}
              <div className="space-y-6">
                <div className="relative">
                  <textarea
                    rows={5}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    placeholder="Enter your message to analyze with AI..."
                    className="w-full backdrop-blur-sm bg-white/10 border border-white/30 rounded-2xl p-4 text-white placeholder-white/60 focus:ring-2 focus:ring-blue-400 focus:outline-none focus:bg-white/20 transition-all duration-300 resize-none"
                  />
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-blue-500/20 to-purple-500/20 opacity-0 hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
                </div>

                <div className="flex gap-4">
                  <button
                    type="submit"
                    className="flex-1 relative overflow-hidden bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white font-bold py-4 rounded-2xl transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:hover:scale-100 shadow-lg hover:shadow-2xl"
                    disabled={loading || !message.trim()}
                  >
                    <div className="relative z-10">
                      {loading ? (
                        <div className="flex items-center justify-center gap-2">
                          <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Analyzing...
                        </div>
                      ) : (
                        "🔍 Analyze Message"
                      )}
                    </div>
                    <div className="absolute inset-0 bg-gradient-to-r from-white/20 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300" />
                  </button>
                  
                  <button
                    type="button"
                    onClick={clearAll}
                    className="flex-1 relative overflow-hidden backdrop-blur-sm bg-white/10 hover:bg-white/20 border border-white/30 text-white font-bold py-4 rounded-2xl transition-all duration-300 transform hover:scale-105 shadow-lg hover:shadow-2xl"
                  >
                    <div className="relative z-10">🧹 Clear</div>
                    <div className="absolute inset-0 bg-gradient-to-r from-white/10 to-transparent opacity-0 hover:opacity-100 transition-opacity duration-300" />
                  </button>
                </div>
              </div>
            </form>

            {/* Results Section with 3D Animation */}
            <div className="space-y-4">
              {error && (
                <div 
                  className="relative bg-red-500/20 backdrop-blur-sm border border-red-400/50 rounded-2xl p-4 text-red-200 font-medium animate-bounce"
                  style={{
                    boxShadow: '0 0 20px rgba(239, 68, 68, 0.3)'
                  }}
                >
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-r from-red-500/10 to-pink-500/10" />
                  <span className="relative z-10">⚠️ Error: {error}</span>
                </div>
              )}

              {result && (
                <div
                  className={`relative overflow-hidden rounded-2xl p-6 text-center font-bold text-lg transition-all duration-500 transform hover:scale-105 ${
                    result.prediction === "spam"
                      ? "bg-gradient-to-r from-red-500/20 to-pink-500/20 text-red-200 border border-red-400/50"
                      : "bg-gradient-to-r from-green-500/20 to-emerald-500/20 text-green-200 border border-green-400/50"
                  }`}
                  style={{
                    boxShadow: result.prediction === "spam" 
                      ? '0 0 30px rgba(239, 68, 68, 0.4)' 
                      : '0 0 30px rgba(34, 197, 94, 0.4)',
                    animation: 'slideInUp 0.6s ease-out'
                  }}
                >
                  <div className={`absolute inset-0 ${
                    result.prediction === "spam" 
                      ? 'bg-gradient-to-r from-red-500/10 to-pink-500/10' 
                      : 'bg-gradient-to-r from-green-500/10 to-emerald-500/10'
                  }`} />
                  
                  <div className="relative z-10">
                    <div className="text-2xl mb-2">
                      {result.prediction === "spam" ? "🚨" : "✅"}
                    </div>
                    <div className="text-3xl mb-2">
                      Prediction: <span className="uppercase font-black">{result.prediction}</span>
                    </div>
                    <div className="text-xl mb-4">
                      Confidence: <span className="font-black">{(result.probability * 100).toFixed(2)}%</span>
                    </div>
                    <div className="text-sm font-normal opacity-80">
                      {result.prediction === "spam"
                        ? "⚠️ This message looks like SPAM"
                        : "✅ This message looks safe (HAM)"}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Custom CSS for animations */}
      <style jsx>{`
        @keyframes slideInUp {
          from {
            opacity: 0;
            transform: translateY(30px) scale(0.9);
          }
          to {
            opacity: 1;
            transform: translateY(0) scale(1);
          }
        }
        
        .animate-slideInUp {
          animation: slideInUp 0.6s ease-out;
        }
      `}</style>
    </div>
  );
}

export default App;