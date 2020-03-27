import React from 'react';
import './App.css';

import {BoxesProvider} from "./context/BoxesContext";
import DocViz from "./components/DocViz";
import {PagesProvider} from "./context/PagesContext";

function App() {
  return (
    <div className="App">
      <BoxesProvider>
        <PagesProvider>
          <DocViz/>
        </PagesProvider>
      </BoxesProvider>
    </div>
  );
}

export default App;
