import React, {useState, createContext} from "react";
import {useInput} from "react-hanger";

export const PagesContext = createContext();

export const PagesProvider = props => {
  const imageFilenameInput = useInput('');
  const scaleInput = useInput(0.13);
  const [pageData, setPageData] = useState('');

  return (
    <PagesContext.Provider value={[imageFilenameInput, scaleInput, pageData, setPageData]}>
      {props.children}
    </PagesContext.Provider>
  );
};
