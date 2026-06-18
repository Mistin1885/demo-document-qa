"use client";

/**
 * MessageList — renders the full conversation thread.
 *
 * Combines server-loaded history + optimistic/streaming messages.
 * Auto-scrolls to the bottom on new messages.
 */

import { useEffect, useRef } from "react";
import type { ChatMessage } from "@/lib/chat/useChatStream";
import { MessageBubble } from "./MessageBubble";

interface MessageListProps {
  messages: ChatMessage[];
}

export function MessageList({ messages }: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center p-8">
        <p className="text-sm text-[var(--muted)] text-center">
          No messages yet. Send a question below.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-4">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
