import React, { useState } from "react";
import ReactDOM from "react-dom";
// import pageImageFile from "./data/doc_01/doc_wrapped_page_0.png";
// import { useFormState } from 'react-use-form-state';

const DocViz = () => {
  const [pageFilename, setPageFilename] = useState('');

  return (
    <div className={"DocViz"}>
      <Controls pageFilename={pageFilename}/>
      <Viewport pageFilename={pageFilename}/>
    </div>
  );
};

const Viewport = (props) => {
  const [filename, setFilename] = useState('');
  React.useEffect(() => {
      setFilename(props.pageFilename);
  }, [props.pageFilename]);

  return <div className="viewport">
    <Page pageFilename={props.pageFilename}></Page>
  </div>
};

const Page = (props) => {
  const [pageFilename, setPageFilename] = useState(props.pageFilename);
  React.useEffect(() => {
      setPageFilename(props.pageFilename);
  }, [props.pageFilename]);


  const dpi = 500;
  const scale = 0.2;
  const pageSize = [8.5, 11];
  const [width, height] = pageSize.map(x => x * dpi * scale);

  const widthStr = `${width}px`;
  const heightStr = `${height}px`;

  const pageStyle = {
    backgroundImage: `url('${pageFilename}')`,
    backgroundSize: `${widthStr} ${heightStr}`,
    backgroundRepeat: "no-repeat",
    width: widthStr,
    height: heightStr,
    // maxHeight: "90%",
    border: "2px solid blue",
  };

  return <div class="page" style={pageStyle}></div>
};


const Controls = (props) => {
  const [filename, setFilename] = useState('');

  return <div className="controls">
    <PageChooser pageFilename={props.pageFilename}/>
    {/*<BoxLoader/>*/}
  </div>
};


const PageChooser = (props) => {
  const [filename, setFilename] = useState(props.pageFilename);
  React.useEffect(() => {
      setFilename(props.pageFilename);
  }, [props.pageFilename]);

  return <div className="page-loader-holder">
    <input
      className="page-file-input"
      type="file"
      name="pageFilenameNameField"
      // placeholder={null}
      onChange = {(e) => {
        console.log("--------event--------");
        console.log(e);
        setFilename(e.target.files[0])
      }}
    />
  </div>
};


ReactDOM.render(
  <DocViz />,
  document.getElementById('root')
);

