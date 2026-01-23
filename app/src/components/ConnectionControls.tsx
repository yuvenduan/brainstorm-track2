'use client';

import { useState, useRef, KeyboardEvent } from 'react';
import { cn } from '@/lib/utils';

interface ConnectionControlsProps {
  isConnected: boolean;
  onConnect: (url: string) => void;
}

export default function ConnectionControls({ isConnected, onConnect }: ConnectionControlsProps) {
  const [serverUrl, setServerUrl] = useState('ws://localhost:8765');
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = () => {
    onConnect(serverUrl);
  };

  const handleKeyPress = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter') {
      handleSubmit();
    }
  };

  return (
    <div className="flex items-center gap-3 justify-center">
      <label htmlFor="server-url" className="text-sm text-text-secondary">
        Server URL:
      </label>
      <input
        ref={inputRef}
        type="text"
        id="server-url"
        value={serverUrl}
        onChange={(e) => setServerUrl(e.target.value)}
        onKeyPress={handleKeyPress}
        className="bg-primary-tertiary border border-border rounded-lg px-3.5 py-2.5 text-text-primary font-mono text-sm w-72 transition-all focus:outline-none focus:border-accent-primary focus:ring-2 focus:ring-accent-glow"
      />
      <button
        onClick={handleSubmit}
        className={cn(
          "bg-accent-primary border-none rounded-lg px-6 py-2.5 text-white font-mono text-sm font-medium cursor-pointer transition-all hover:bg-[#6b5cd4] hover:-translate-y-0.5 hover:shadow-lg hover:shadow-accent-glow active:translate-y-0 disabled:opacity-50 disabled:cursor-not-allowed",
          isConnected && "bg-error hover:bg-[#e85656]"
        )}
      >
        {isConnected ? 'Disconnect' : 'Connect'}
      </button>
    </div>
  );
}