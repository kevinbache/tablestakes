import React, {useState} from 'react';
import ReactDOM from "react-dom";
import Papa from 'papaparse';

import selDataFile from '../data/doc_01/sel_words.csv';
import ocrDataFile from '../data/doc_01/ocr_words.csv';
import imageFile from '../data/doc_01/doc_wrapped_page_0.png';

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

function scale_ranges(ranges, xOffset, xScale, yOffset, yScale) {
  // ranges should be an object like the one produced by bbox_to_ranges:
  //    [xmin, xmax, ymin, ymax]
  return {
    xmin: parseFloat(ranges[0] * xScale + xOffset),
    xmax: parseFloat(ranges[1] * xScale + xOffset),
    ymin: parseFloat(ranges[2] * yScale + yOffset),
    ymax: parseFloat(ranges[3] * yScale + yOffset),
  }
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
      dpi: 500,
      scale: 0.3,
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
    const extraWidth = 6;
    const extraHeight = 6;
    const yOffset = 4;
    const xOffset = 4;
    const box_style = {
      left: datum.xmin + xOffset,
      top: datum.ymin + yOffset,
      width: (datum.xmax - datum.xmin) * inflation_factor + extraWidth,
      height: (datum.ymax - datum.ymin) * inflation_factor + extraHeight,
      backgroundColor: 'rgba(60, 60, 240, 0.2)',
      border: "1px solid blue",
      position: 'absolute',
    };
    return (
      <div
        style={box_style}
        key={index}
        wordtext={datum.text}
      >
      </div>
    );
  }

  render() {
    const image_size = [8.5, 11];

    const [width, height] = image_size.map(x => x * this.dpi * this.scale) ;

    const widthStr = `${width}px`;
    const heightStr = `${height}px`;

    const bg_div_style = {
      backgroundImage: `url('${imageFile}')`,
      backgroundSize: `${widthStr} ${heightStr}`,
      backgroundRepeat: "no-repeat",
      width: widthStr,
      height: heightStr,
      // maxHeight: "90%",
      border: "0px solid blue",
    };

    /*
    I want to say something like:
      <Page img_src=importedImageFile>
        <Boxes csvFile=importedSelCsvFile color='rgba(60, 240, 60, 0.2)'>
        <Boxes csvFile=importedOcrCsvFile color='rgba(60, 60, 240, 0.2)'>
      </Page>

      there's a directory chooser component which selects the importedFiles and creates the boxes
    */

    return (
      <div className="page-holder" style={bg_div_style}>
        { this.state.ocr_data.map((datum, index) => this.datum_2_box(datum, index)) }
      </div>
    );
  }
}

// resizable?
//    https://stackoverflow.com/questions/38836553/how-to-use-jquery-ui-with-react-js
class Boxes extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      wordData: [],
    }
  }

  render() {
    return null;
  }

}

ReactDOM.render(
  <Page />,
  document.getElementById('root')
);

