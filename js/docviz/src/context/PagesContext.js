import React, {useState, createContext} from "react";

export const PagesContext = createContext();

export const PagesProvider = props => {
  const [state, setState] = useState([]);

  return (
    <PagesContext.Provider value={[state, setState]}>
      {props.children}
    </PagesContext.Provider>
  );
};
