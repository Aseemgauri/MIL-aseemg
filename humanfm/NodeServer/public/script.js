"use strict";

(function() {
  window.addEventListener("load", init);

  let simulationTimerId;
  let deletionTimerId;
  const CLASS_DATA_URL = "/HumanFM/classes";
  const LEVEL_DATA_URL = "/HumanFM/classes/update";

  // The initial function to set-up the program when the window loads.
  function init() {
    let startBtn = id("start-btn");
    let endBtn = id("end-btn");
    let resetBtn = id("reset-btn");

    startBtn.addEventListener("click", function() {
      startSimulation();
    });
    endBtn.addEventListener("click", function() {
      stopSimulation();
    });
    resetBtn.addEventListener("click", function() {
      resetSimulation();
    });
  }

  /**
   * Responsible for beginning the simulation on the user side (displays certain classes
   * and then removes them).
   */
  function startSimulation() {
    id("start-btn").disabled = true;
    simulationTimerId = setInterval(function() {
      fetchClassData();
    }, 5000);
  }

  /**
   * Stops the simulation by preventing new data from being fetched.
   */
  function stopSimulation() {
    id("start-btn").disabled = false;
    if (simulationTimerId) {
      clearInterval(simulationTimerId);
      simulationTimerId = null;
    }
  }

  /**
   * Resets the audio dashboard and removes all present sounds.
   */
  function resetSimulation() {
    id("start-btn").disabled = false;
    let classCtnrElmts = id("class-ctnr");

    while (classCtnrElmts.firstElementChild) {
      classCtnrElmts.removeChild(classCtnrElmts.firstElementChild);
    }
  }

  /**
   * Helper function to remove a random classes from view.
   */
  // Comment out random class removal functionality
  /*
  function removeRandomClass() {
    deletionTimerId = setInterval(function() {
      let randDelay = (Math.floor(Math.random() * 15 + 10)) * 1000;

      setTimeout(async function() {
        let totalChildren = id("class-ctnr").children.length - 1;
        if (totalChildren >= 0 && totalChildren < id("class-ctnr").children.length) {
          let randIndex = Math.floor(Math.random() * totalChildren);
          id("class-ctnr").removeChild(id("class-ctnr").children[randIndex]);
        }
      }, randDelay)
    }, 5000);
  }
  */

  /**
   * Helper function to find a specific class id on the user interface.
   * @param {int} currId - The id that we are searching for.
   * @returns - A boolean value.
   */
  function findClassId(currId) {
    let displayedClasses = id("class-ctnr").children;
    for (let i = 0; i < displayedClasses.length; i++) {
      if (displayedClasses[i].id == "class-" + currId) {
        return true;
      }
    }
    return false;
  }

  /**
   * Fetch function to extract data associated with a random class from the server.
   */
  async function fetchClassData() {
    try {
      console.log("Fetching classification data from server...");
      let randClass = await fetch(CLASS_DATA_URL);
      await statusCheck(randClass);
      let randClassData = await randClass.json();
      console.log("Received data from server:", randClassData);
      processClassData(randClassData);
      
      if (randClassData.length > 0) {
        console.log(`Displaying ${randClassData.length} classifications from server`);
      } else {
        console.log("No classifications received from server");
      }
    } catch (err) {
      handleError(err);
    }
  }

  /**
   * Helper to function to create UI elements associated with JSON from server.
   * @param {Object} randClassData - JSON object from server.
   */
  function processClassData(randClassData) {
    // Clear existing classes first to show current state
    let classCtnrElmts = id("class-ctnr");
    while (classCtnrElmts.firstElementChild) {
      classCtnrElmts.removeChild(classCtnrElmts.firstElementChild);
    }
    
    // Display all detected classes
    for (let classData of randClassData) {
      let parentCtnr = id("class-ctnr");
      let classCtnr = gen("article");
      let classBtn = gen("button");
      let classInput = gen("input");

      classCtnr.id = "class-" + classData.id;
      classBtn.textContent = classData.name;
      classBtn.classList.add("active"); // Mark as active since it's detected
      classBtn.addEventListener("click", function() {
        classBtn.classList.toggle("selected");
      });
      classInput.type = "range";
      classInput.min = 0;
      classInput.max = 100;
      classInput.value = classData.level; // Use the actual level from database
      classInput.disabled = false; // Enable slider so user can adjust volume
      classInput.addEventListener("input", async function(event) {
        await fetchClassLevel(event)
      });

      classCtnr.appendChild(classBtn);
      classCtnr.appendChild(classInput);
      parentCtnr.appendChild(classCtnr);
    }
    
    // Log what's being displayed
    if (randClassData.length > 0) {
      console.log(`Displaying ${randClassData.length} active classifications:`, 
        randClassData.map(c => c.name).join(", "));
    } else {
      console.log("No active classifications detected");
    }
  }

  /**
   * Fetch function to update the server with new audio level data.
   * @param {Event} event - A JS event associated with the audio input tag.
   */
  async function fetchClassLevel(event) {
    try {
      let classData = new FormData();
      classData.append("id", event.target.parentElement.id.substring(6))
      classData.append("level", event.target.value)
      let updateStat = await fetch(LEVEL_DATA_URL, {method: "POST", body: classData});
      await statusCheck(updateStat);
      let updateStatData = await updateStat.text();
    } catch (err) {
      handleError(err);
    }
  }

  /**
   * An function responsible for handling errors for various server interacting functions.
   * @param {*} err - The actual error that occurred.
   */
  function handleError(err) {
    console.log("Oops! There has been a error fetching data");
    console.error(err);
  }

  /**
   * This is a helper function that allows one to check if the network request that was
   * made to the API ended up being successful or not based on a certain status code. If
   * the the request is not successful, then the appropriate error handling functionality
   * is employed.
   * @param {Response} res - This is a response object representing the response sent
   *                         by the server in regards to the information requested by
   *                         the specific API.
   * @returns {Response} - If the response comes back as successful, then the original
   *                       response object is returned back.
   */
  async function statusCheck(res) {
    if (!res.ok) {
      throw new Error(await res.text());
    }
    return res;
  }

  /**
   * This is a helper function that allows for a shorthand version of creating a new
   * element for the DOM tree.
   * @param {string} tag - This is a string representation of the desired HTML tag.
   * @return {DOMElement} - This function returns a brand new HTML element.
   */
  function gen(tag) {
    return document.createElement(tag);
  }

  /**
   * This is a helper function that allows for a shorthand version of searching for
   * an HTML element based on its id.
   * @param {string} id - This is a string representation of the id for the desired element.
   * @return {DOMElement} - This function returns the desired HTML element if it was found.
   */
  function id(id) {
    return document.getElementById(id);
  }
})();