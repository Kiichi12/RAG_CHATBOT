import { useState, useEffect } from 'react'
import axios from "axios"
import "./App.css"

function App() {
  const [inputMessage, setInputMessage] = useState("");
  const [messages, setMessages] = useState([{ id: 1, text: "Hello! How can I help you?", sender: "bot" }]);
  const [isLoading, setIsLoading] = useState(false);
  const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://localhost:5000"; // Default to localhost if not set
  // axios.post(`${BACKEND_URL}/api/init`)
  // useEffect(() => {
  //   const initBackend = async () => {
  //     try {
  //       await axios.post(`${BACKEND_URL}/api/init`);
        
  //     } catch(error) {
  //       console.error("Initialization error:", error);
  //     }
  //   };
  //   initBackend();
  // }, []);
 useEffect(() => {
  (async () => {
    try {
      await axios.post(`${BACKEND_URL}/api/init`);
    } catch (err) {
      console.error("Initialization error:", err);
    }
  })();
}, []); // Empty dependency array ensures it runs only once


  const handleSend = async () => {
    if(!inputMessage.trim()) return;
    console.log(inputMessage)
    
    const userMessage = { id: Date.now(), text: inputMessage, sender: "user" };
    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);
    
    try {
      const response = await axios.post(`${BACKEND_URL}/api/sendprompt`,userMessage);
      
      if(response.status === 200) {
        const botMessage = {
          id: Date.now() + 1,
          text: response.data.answer || response.data.response,
          sender: "bot"
        };
        setMessages(prev => [...prev, botMessage]);
      } 
      else 
      {
        console.log("failed send prompt:"+response)
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I'm having trouble connecting to the server.",
        sender: "bot"
      };
      setMessages(prev => [...prev, errorMessage]);
      console.error("API Error:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <p className='page-desc'>
        This is a chatbot designed to answer any questions you have about INFINTUM 2025
      </p>
      <div className='input-area'>
        <input 
          type='text'
          placeholder='Enter your prompt here'
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleSend()}
          disabled={isLoading}
        />
        <button onClick={handleSend} disabled={isLoading}>
          {isLoading ? "Sending..." : "Send"}
        </button>
      </div>
      <div className="messages">
        {messages.map(msg => (
          <div 
            key={msg.id}
            className={`message ${msg.sender === "user" ? "user-message" : "bot-message"}`}
          >
            {msg.text}
          </div>
        ))}
        {isLoading && (
          <div className="message bot-message">Thinking...</div>
        )}
      </div>
      
    </div>
  );
}

export default App;