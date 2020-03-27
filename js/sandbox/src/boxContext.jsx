import React, {useState, createContext} from "react";

export const BoxesContext = createContext();

export const BoxesProvider = props => {
  const [state, setState] = useState({
    boxes: [],
    filename: [],
    xScale: 1.0,
    xOffset: 0.0,
    yScale: 1.0,
    yOffset: 0.0,
  });

  return (
    <BoxesContext.Provider value={[state, setState]}>
      {props.children}
    </BoxesContext.Provider>
  );
};
