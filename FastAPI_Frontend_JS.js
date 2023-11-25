function handleCpuSubmit(event) {
  let userPrompt = document.getElementById("user_prompt");
  let chatWithCpuId = document.getElementById("chatWithCpuId");

  fetch("http://127.0.0.1:8000/prompt_route", {
    method: "POST",
    body: JSON.stringify({ user_prompt: userPrompt.value }),
    headers: {
      "Content-Type": "application/json",
    },
  })
    .then(async (response) => {
      console.log("Success");
      const data = await response.json();
      console.log(data);
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}
