import React from 'react';
import ReactDOM from "react-dom";
import Papa from 'papaparse';

import selDataFile from './data/doc_1/sel_data.csv';
import ocrDataFile from './data/doc_1/ocr_data.csv';
import imageFile from './data/doc_1/page_01.png';

function bbox_to_ranges(bbox) {
  // bbox should be a string like "BBox(x=[1947, 2149], y=[1234, 1284])"
  let xre = /x=\[(?<xmin>\d+), (?<xmax>\d+)\]/;
  let yre = /y=\[(?<ymin>\d+), (?<ymax>\d+)\]/;
  const xmatch = bbox.match(xre);
  const ymatch = bbox.match(yre);

  return {
    xmin: parseFloat(xmatch.groups.xmin),
    xmax: parseFloat(xmatch.groups.xmax),
    ymin: parseFloat(ymatch.groups.ymin),
    ymax: parseFloat(ymatch.groups.ymax),
  };
}

function sel_row_to_ranges(row) {
  return {
    xmin: parseFloat(row.left),
    xmax: parseFloat(row.right),
    ymin: parseFloat(row.bottom),
    ymax: parseFloat(row.top),
  }
}

class Page extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      sel_data: [],
      ocr_data: [],
    };
  }

  componentDidMount() {

    Papa.parse(selDataFile, {
        download: true,
        header: true,
        step: row => {
          // console.log("sel row: ");
          // console.log(row);
          const data = row.data;
          if ( !(data.text) ) { return null; }

          const ranges = sel_row_to_ranges(data);
          const newData = {
            text: data.text,
            ...ranges,
          };
          this.setState((prevState) => ({ ocr_data: prevState.sel_data.concat([newData])}));
        },
    });

    Papa.parse(ocrDataFile, {
        download: true,
        header: true,
        step: row => {
          // console.log("ocr row: ");
          // console.log(row);
          let data = row.data;
          if (data.word_type === "WordType.LINEBREAK" || !('bbox' in data)) { return null; }

          const ranges = bbox_to_ranges(data.bbox);
          const newData = {
            text: data.text,
            ...ranges,
          };
          // TODO: this is n^2 in number of words
          this.setState((prevState) => ({ ocr_data: prevState.ocr_data.concat([newData])}));
        },
    });
  }

  datum_2_box(datum, index) {
    // TODO: inflation_factor and yOffset
    const inflation_factor = 1.0;
    const extraWidth = 5;
    const extraHeight = 5;
    const yOffset = 3;
    const xOffset = 5;
    const box_style = {
      left: datum.xmin + xOffset,
      top: datum.ymin + yOffset,
      width: (datum.xmax - datum.xmin) * inflation_factor + extraWidth,
      height: (datum.ymax - datum.ymin) * inflation_factor + extraHeight,
      backgroundColor: 'rgba(20, 20, 240, 0.2)',
      border: "1px solid blue",
      position: 'absolute',
    };
    return (
      <div
        style={box_style}
        key={index}
        datawordtext={datum.text}
      >
      </div>
    );
  }

  render() {
    const widthStr = `${8.5 * 500}px`;
    const heightStr =  `${11 * 500}px`;
    const bg_div_style = {
      backgroundImage: `url('${imageFile}')`,
      backgroundSize: `${widthStr} ${heightStr}`,
      backgroundRepeat: "no-repeat",
      width: widthStr,
      height: heightStr,
      border: "0px solid blue",
    };

    return <div className="page-holder" style={bg_div_style}>
      { this.state.ocr_data.map((datum, index) => this.datum_2_box(datum, index)) }
    </div>;
  }
}

// class Boxes extends React.Component {
//
// }

ReactDOM.render(
  <Page />,
  document.getElementById('root')
);
