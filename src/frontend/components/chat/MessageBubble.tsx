"use client";

/**
 * MessageBubble — renders one message (user, assistant, or in-flight streaming).
 *
 * Features:
 *  - Role-based visual style (user right-aligned, assistant left-aligned).
 *  - Citation chips row below assistant content.
 *  - Debug trace collapsed drawer for assistant messages.
 *  - Streaming indicator while status="streaming".
 *  - Error / stopped state display.
 */

import type { MessageRead } from "@/lib/api/types";
import type { OptimisticUserMessage, StreamingMessage } from "@/lib/chat/useChatStream";
import { CitationChip } from "./CitationChip";
import { DebugTraceDrawer } from "./DebugTraceDrawer";

type AnyMessage = MessageRead | OptimisticUserMessage | StreamingMessage;

function isStreamingMessage(m: AnyMessage): m is StreamingMessage {
  return m.role === "assistant" && "partial" in m;
}

function isOptimisticUser(m: AnyMessage): m is OptimisticUserMessage {
  return m.role === "user" && !("content" in (m as MessageRead));
}

function isMessageRead(m: AnyMessage): m is MessageRead {
  return "session_id" in m;
}

interface MessageBubbleProps {
  message: AnyMessage;
}

export function MessageBubble({ message }: MessageBubbleProps) {
  const isUser = message.role === "user";
  const isStreaming = isStreamingMessage(message) && message.status === "streaming";
  const isStopped = isStreamingMessage(message) && message.status === "stopped";
  const isError = isStreamingMessage(message) && message.status === "error";

  // Content text
  let content = "";
  if (isMessageRead(message)) {
    content = message.content;
  } else if (isOptimisticUser(message)) {
    content = (message as OptimisticUserMessage).content;
  } else if (isStreamingMessage(message)) {
    content = message.partial;
  }

  // Citations
  const citations =
    isMessageRead(message)
      ? (message.citations ?? [])
      : isStreamingMessage(message)
      ? message.citations
      : [];

  // Tool trace (from server message or done streaming message)
  const toolTrace =
    isMessageRead(message)
      ? message.tool_trace
      : isStreamingMessage(message) && message.finalResponse
      ? message.finalResponse.debug_trace
      : null;

  return (
    <div
      className={`flex w-full mb-4 ${isUser ? "justify-end" : "justify-start"}`}
    >
      <div
        className={`max-w-[75%] ${
          isUser
            ? "bg-[var(--accent)] text-white rounded-2xl rounded-br-sm px-4 py-2.5"
            : "bg-[var(--surface-raised)] text-[var(--foreground)] rounded-2xl rounded-bl-sm px-4 py-2.5"
        }`}
      >
        {/* Main content */}
        <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {content}
          {isStreaming && (
            <span className="inline-flex gap-1 ml-1 align-middle">
              <span className="w-1 h-1 rounded-full bg-current opacity-60 animate-bounce [animation-delay:0ms]" />
              <span className="w-1 h-1 rounded-full bg-current opacity-60 animate-bounce [animation-delay:150ms]" />
              <span className="w-1 h-1 rounded-full bg-current opacity-60 animate-bounce [animation-delay:300ms]" />
            </span>
          )}
          {!content && isStreaming && (
            <span className="text-xs opacity-60">Planning and retrieving evidence…</span>
          )}
        </p>

        {/* Stopped / error banners */}
        {isStopped && (
          <p className="mt-1 text-xs opacity-60 italic">Generation stopped.</p>
        )}
        {isError && (
          <p className="mt-1 text-xs text-rose-300 italic">
            {isStreamingMessage(message) ? message.errorDetail : "Error"}
          </p>
        )}

        {/* Citation chips */}
        {citations.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-1.5">
            {citations.map((c, i) => (
              <CitationChip key={c.citation_id} citation={c} index={i} />
            ))}
          </div>
        )}

        {/* Debug trace */}
        {!isUser && toolTrace && <DebugTraceDrawer trace={toolTrace} />}
      </div>
    </div>
  );
}
