import React, {useState, createContext} from "react";
import {useInput} from "react-hanger";

export const PagesContext = createContext();

export const PagesProvider = props => {
  const imageFilenameInput = useInput('');
  const scaleInput = useInput(0.2);

  return (
    <PagesContext.Provider value={[imageFilenameInput, scaleInput]}>
      {props.children}
    </PagesContext.Provider>
  );
};
