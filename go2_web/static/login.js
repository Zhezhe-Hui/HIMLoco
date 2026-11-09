"use strict";
document.getElementById("loginForm").addEventListener("submit",async event=>{
  event.preventDefault();
  const button=document.getElementById("loginButton"),message=document.getElementById("loginMessage");
  button.disabled=true;message.textContent="";
  try{
    const response=await fetch("/api/auth/login",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({username:document.getElementById("username").value,password:document.getElementById("password").value})});
    const result=await response.json();
    if(!response.ok)throw new Error(result.error||"登录失败");
    document.getElementById("password").value="";
    location.replace("/");
  }catch(error){message.textContent=error.message;button.disabled=false;}
});
