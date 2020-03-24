import React, { useState } from "react";
import Papa from "papaparse";

const Upload = () => {
  const [state, setState] = useState({
    csvfile: null,
    event_target_files: null,
    data: null,
  });

  function handleChange(event) {
    setState({
      csvfile: event.target.files[0],
      event_target_files: event.target.files,
    });
  }

  function importCSV(e) {
    e.preventDefault();
    const { csvfile } = state;

    csvfile &&
      Papa.parse(csvfile, {
        header: true,
        dynamicTyping: true,
        complete: setDataInState
      });
  }

  function setDataInState(result) {
    var data = result.data;
    setState({
      state_bak: this.state,
      data: data,
    });
    console.log(data);
  }

  return (
    <div>
      <div className="upload">
        <p>Paragraph upload paragraph</p>
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
