"use client";
import { useState } from 'react';
import axios from 'axios';

export default function Home() {
  const [files, setFiles] = useState<FileList | null>(null);
  const [message, setMessage] = useState('');
  const [chatHistory, setChatHistory] = useState<{ role: string; content: string }[]>([]);
  const [loading, setLoading] = useState(false);

  const handleUpload = async () => {
    if (!files) return;
    const formData = new FormData();
    Array.from(files).forEach(file => formData.append('files', file));
    setLoading(true);
    try {
      await axios.post(`${process.env.NEXT_PUBLIC_BASE_URL}/upload-pdfs`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      alert('PDFs indexed!');
    } catch (error) {
      alert('Upload failed');
    }
    setLoading(false);
  };

  const handleSend = async () => {
    if (!message) return;
    setChatHistory(prev => [...prev, { role: 'user', content: message }]);
    setLoading(true);
    try {
      const res = await axios.post(`${process.env.NEXT_PUBLIC_BASE_URL}/chat`, { message });
      setChatHistory(prev => [...prev, { role: 'assistant', content: res.data.response }]);
    } catch (error) {
      setChatHistory(prev => [...prev, { role: 'assistant', content: 'Error occurred' }]);
    }
    setMessage('');
    setLoading(false);
  };

  return (
    <div className="p-8 max-w-2xl mx-auto">
      <h1 className="text-2xl font-bold mb-4">Multi-PDF RAG Chat Agent</h1>
      
      {/* Upload */}
      <div className="mb-4">
        <input type="file" multiple accept=".pdf" onChange={e => setFiles(e.target.files)} className="mr-2" />
        <button onClick={handleUpload} disabled={loading} className="bg-blue-500 text-white px-4 py-2 rounded">
          Upload & Index PDFs
        </button>
      </div>
      
      {/* Chat */}
      <div className="border p-4 h-96 overflow-y-auto mb-4">
        {chatHistory.map((msg, i) => (
          <div key={i} className={`mb-2 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
            <span className={`inline-block p-2 rounded ${msg.role === 'user' ? 'bg-blue-200' : 'bg-gray-200'}`}>
              {msg.content}
            </span>
          </div>
        ))}
        {loading && <div>Thinking...</div>}
      </div>
      
      <div className="flex">
        <input
          type="text"
          value={message}
          onChange={e => setMessage(e.target.value)}
          onKeyPress={e => e.key === 'Enter' && handleSend()}
          className="flex-1 border p-2 mr-2"
          placeholder="Ask about your PDFs..."
        />
        <button onClick={handleSend} disabled={loading} className="bg-green-500 text-white px-4 py-2 rounded">
          Send
        </button>
      </div>
    </div>
  );
}
