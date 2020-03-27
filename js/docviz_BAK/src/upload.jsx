import React, { useState } from "react";
import Papa from "papaparse";

function Example(props) {
  // You can use Hooks here!
  return <div />;
}

const Upload = () => {
  const [state, setState] = useState({
    filename: null,
    data: null,
  });

  // ////////////////////////////////////////////////////////////////////////////////////
  // // https://stackoverflow.com/questions/38836553/how-to-use-jquery-ui-with-react-js
  // let defaultProps = {
  //   enable: true
  // };
  //
  // // Optional: set the prop types
  // let propTypes = {
  //   enable: React.PropTypes.bool,
  //   // handleData: React.PropTypes.func.isRequired
  // };
  // ////////////////////////////////////////////////////////////////////////////////////

  function handleChange(event) {
    setState({
      filename: event.target.files[0],
    });
  }

  // function importCSV(e) {
  //   e.preventDefault();
  //   const { csvfile } = state;
  //
  //   if (csvfile) {
  //     Papa.parse(csvfile, {
  //       header: true,
  //       dynamicTyping: true,
  //         complete: setDataInState
  //     });
  //   }
  // }

  // function setDataInState(result) {
  //   var data = result.data;
  //   setState({
  //     state_bak: this.state,
  //     data: data,
  //   });
  //   console.log(data);
  // }

  return (
    <div>
      <div className="upload">
        <p>Paragraph upload paragraph</p>

        {/*
        directory selector:
          https://stackoverflow.com/questions/12942436/how-to-get-folder-directory-from-html-input-type-file-or-any-other-way
            <input type="file" id="ctrl" webkitdirectory directory multiple/>
          https://developer.mozilla.org/en-US/docs/Web/API/HTMLInputElement/webkitdirectory
            <input type="file" id="ctrl" webkitdirectory="true" directory multiple/> ?
        */}
        <input
          className="upload-input"
          type="file"
          name="upload_file"
          placeholder={null}
          onChange={handleChange}
        />
        <p />
        <button onClick={importCSV}>Upload</button>
      </div>
    </div>
  );
};

export default Upload;
