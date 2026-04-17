import { useEffect } from 'react';
import CuriaApp from './CuriaApp.jsx';
import './Curia.css';

export default function Curia() {
  useEffect(() => {
    document.body.classList.add('curia-page');
    return () => document.body.classList.remove('curia-page');
  }, []);

  return (
    <div className="curia-app">
      <CuriaApp />
    </div>
  );
}
