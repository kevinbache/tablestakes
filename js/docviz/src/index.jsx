import React, { useState } from "react";
import ReactDOM from "react-dom";

const Page = () => {
  const [state, setState] = useState({
    filename: null,
    data: null,
  });

  return <div />
};


ReactDOM.render(
  <Page />,
  document.getElementById('root')
);

