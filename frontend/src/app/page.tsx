"use client";

import React, { useState } from "react";
import { Upload, Send, FileText, CheckCircle2, Crosshair, AlertCircle } from "lucide-react";

interface BBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

interface Citation {
  id: string;
  page: number;
  bbox: BBox;
  content_preview: string;
  confidence: number;
}

interface QueryResponse {
  query: string;
  answer: string;
  citations: Citation[];
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [uploadedFilename, setUploadedFilename] = useState<string>("");
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [query, setQuery] = useState<string>("");
  const [isQuerying, setIsQuerying] = useState<boolean>(false);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [selectedCitation, setSelectedCitation] = useState<Citation | null>(null);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.[0]) return;
    const selectedFile = e.target.files[0];
    setFile(selectedFile);
    setIsUploading(true);
    setErrorMessage(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/upload", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadedFilename(data.filename);
        setResult(null);
        setSelectedCitation(null);
      } else {
        setErrorMessage(data.detail || "Upload failed");
      }
    } catch (err) {
      setErrorMessage("Could not connect to the backend server on http://127.0.0.1:8000");
    } finally {
      setIsUploading(false);
    }
  };

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setIsQuerying(true);
    setErrorMessage(null);

    try {
      const res = await fetch("http://127.0.0.1:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      const data = await res.json();
      if (res.ok) {
        setResult(data);
        if (data.citations && data.citations.length > 0) {
          setSelectedCitation(data.citations[0]);
        }
      } else {
        setErrorMessage(data.detail || "Query failed");
      }
    } catch (err) {
      setErrorMessage("Failed to query knowledge agent backend.");
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Left Column: Chat & Agent Reasoning */}
      <div className="w-1/2 flex flex-col border-r border-slate-800 p-6 space-y-6 overflow-y-auto">
        <header className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              <span className="p-1.5 bg-blue-600 rounded-lg text-white">
                <FileText className="w-5 h-5" />
              </span>
              Multimodal Enterprise Agent
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Hybrid Search (BM25 + Dense) & Neural Cross-Encoder Grounding
            </p>
          </div>

          {/* Document Upload Button */}
          <div>
            <label className="cursor-pointer inline-flex items-center gap-2 px-3.5 py-2 rounded-md bg-slate-800 hover:bg-slate-700 text-xs font-medium border border-slate-700 transition">
              <Upload className="w-3.5 h-3.5 text-blue-400" />
              {isUploading ? "Indexing Chunks..." : uploadedFilename ? "Replace PDF" : "Upload Document"}
              <input type="file" accept=".pdf" className="hidden" onChange={handleUpload} />
            </label>
          </div>
        </header>

        {errorMessage && (
          <div className="flex items-center gap-2 px-3 py-2 bg-red-950/50 border border-red-800/80 rounded-md text-xs text-red-300">
            <AlertCircle className="w-4 h-4 shrink-0 text-red-400" />
            <span>{errorMessage}</span>
          </div>
        )}

        {uploadedFilename && (
          <div className="flex items-center gap-2 px-3 py-2 bg-slate-900 border border-slate-800 rounded-md text-xs text-emerald-400">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>Active Document: <strong>{uploadedFilename}</strong> (indexed)</span>
          </div>
        )}

        {/* Query Input */}
        <form onSubmit={handleQuery} className="flex gap-2">
          <input
            type="text"
            placeholder={uploadedFilename ? "Ask anything about the document..." : "Upload a PDF first to ask questions"}
            disabled={!uploadedFilename || isQuerying}
            className="flex-1 bg-slate-900 border border-slate-800 rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-blue-500 disabled:opacity-50"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <button
            type="submit"
            disabled={isQuerying || !uploadedFilename || !query.trim()}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-lg text-sm font-medium flex items-center gap-1.5 transition"
          >
            <Send className="w-4 h-4" />
            {isQuerying ? "Reasoning..." : "Send"}
          </button>
        </form>

        {/* Agent Response Feed */}
        {result && (
          <div className="bg-slate-900 border border-slate-800 rounded-lg p-5 space-y-4">
            <h3 className="text-sm font-semibold text-slate-300">Agent Reasoning Output:</h3>
            <p className="text-sm whitespace-pre-line text-slate-200 leading-relaxed">
              {result.answer}
            </p>

            {/* Citations List */}
            {result.citations.length > 0 && (
              <div className="pt-4 border-t border-slate-800">
                <span className="text-xs font-semibold uppercase text-slate-400 tracking-wider">
                  Retrieved Citations (Click to Ground on Page):
                </span>
                <div className="mt-2.5 space-y-2">
                  {result.citations.map((c) => (
                    <button
                      key={c.id}
                      onClick={() => setSelectedCitation(c)}
                      className={`w-full text-left p-2.5 rounded-md text-xs border transition flex items-start justify-between ${
                        selectedCitation?.id === c.id
                          ? "bg-blue-950/60 border-blue-500 text-blue-200"
                          : "bg-slate-950 border-slate-800 text-slate-400 hover:border-slate-700"
                      }`}
                    >
                      <div className="pr-2">
                        <span className="font-bold text-white mr-2">[{c.id}] Page {c.page}</span>
                        <span className="italic">{c.content_preview}</span>
                      </div>
                      <span className="text-[10px] bg-slate-800 px-1.5 py-0.5 rounded text-slate-300 shrink-0">
                        score: {c.confidence.toFixed(2)}
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Column: Visual Citation Grounding Inspector */}
      <div className="w-1/2 flex flex-col p-6 bg-slate-900/50">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center gap-2 text-sm font-semibold text-slate-200">
            <Crosshair className="w-4 h-4 text-blue-400" />
            <span>Visual Grounding Preview</span>
          </div>
          {selectedCitation && (
            <span className="text-xs text-slate-400">
              Page {selectedCitation.page} • Citation #{selectedCitation.id}
            </span>
          )}
        </div>

        <div className="flex-1 flex items-center justify-center p-4">
          {uploadedFilename && selectedCitation ? (
            <div className="relative border border-slate-700 bg-white rounded-lg shadow-2xl overflow-hidden max-h-[85vh] aspect-[1/1.3] w-full max-w-[500px]">
              {/* Actual PDF Page Image rendered by FastAPI PyMuPDF */}
              <img
                src={`http://127.0.0.1:8000/api/document/${uploadedFilename}/page/${selectedCitation.page}`}
                alt={`Page ${selectedCitation.page}`}
                className="absolute inset-0 w-full h-full object-contain pointer-events-none select-none"
              />

              {/* Dynamic Bounding Box Highlight Overlay */}
              <div
                className="absolute border-2 border-red-500 bg-red-500/25 rounded transition-all duration-300 ease-out pointer-events-none"
                style={{
                  left: `${selectedCitation.bbox.x1 * 100}%`,
                  top: `${selectedCitation.bbox.y1 * 100}%`,
                  width: `${Math.max((selectedCitation.bbox.x2 - selectedCitation.bbox.x1) * 100, 4)}%`,
                  height: `${Math.max((selectedCitation.bbox.y2 - selectedCitation.bbox.y1) * 100, 3)}%`,
                }}
              >
                <span className="absolute -top-5 left-0 bg-red-600 text-white text-[10px] px-1 rounded font-semibold uppercase shadow">
                  {selectedCitation.id}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-center text-slate-500 text-sm max-w-xs leading-relaxed">
              Upload a PDF document and submit a question to inspect visual bounding box citations.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}