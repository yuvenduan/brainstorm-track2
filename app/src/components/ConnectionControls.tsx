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

  return (
    <div className="flex items-center gap-3 justify-center">
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