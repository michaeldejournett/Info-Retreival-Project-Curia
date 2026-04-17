import { useEffect, useRef, useState } from 'react';
import { getMessages, postMessage, isMember, getUser } from '../utils/groups.js';

function formatTime(ts) {
  return new Date(ts).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' });
}

export default function GroupMessageBoard({ eventId, group }) {
  const user = getUser();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const listRef = useRef(null);
  const member = isMember(group);

  useEffect(() => {
    setMessages(getMessages(eventId, group.id));
    const interval = setInterval(() => {
      setMessages(getMessages(eventId, group.id));
    }, 3000);
    return () => clearInterval(interval);
  }, [eventId, group.id]);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = listRef.current.scrollHeight;
  }, [messages]);

  const handleSend = () => {
    if (!input.trim() || !member) return;
    postMessage(eventId, group.id, input.trim());
    setMessages(getMessages(eventId, group.id));
    setInput('');
  };

  return (
    <div className="msg-board">
      <div className="msg-list" ref={listRef}>
        {messages.length === 0 && <p className="msg-empty">No messages yet — say hello!</p>}
        {messages.map((msg, i) => {
          const mine = msg.authorId === user.id;
          return (
            <div key={i} className={`msg ${mine ? 'msg-mine' : ''}`}>
              <div className="msg-header">
                <span className="msg-author">{msg.authorName}</span>
                <span className="msg-time">{formatTime(msg.ts)}</span>
              </div>
              <div className="msg-body">{msg.body}</div>
            </div>
          );
        })}
      </div>

      {member ? (
        <div className="msg-input-row">
          <input
            className="msg-input"
            type="text"
            placeholder="Type a message…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          />
          <button className="btn btn-primary btn-sm" onClick={handleSend} disabled={!input.trim()}>
            Send
          </button>
        </div>
      ) : (
        <p className="msg-join-hint">Join the group to send messages</p>
      )}
    </div>
  );
}
